from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form,status,Body, Query
from sqlalchemy.orm import Session, joinedload
from sqlalchemy import func
from app.schemas.project import ProjectCreate, ProjectResponse, ProjectCheckResponse, FileDeleteItem, QueryRequest, QuerySessionOut, MessageOut, UpdateLLMConfigInput, UserOut
from app.services.project_service import get_project, create_project, process_uploaded_file,get_all_projects,get_deleted_projects,get_username_from_user_id,get_project_active
from app.db.session import get_db, get_files_db, get_websocket_db
from datetime import date,datetime,timezone
from typing import Optional
from typing import List, Dict
from fastapi import Query
import os
import logging
import time
from app.core.security import azure_ad_dependency, websocket_auth
from app.models.user import User ,Project, ClinicalQuerySession , ClinicalQueryMessage, LLMProvider, LLMModel, UserLLMConfig,UploadBatch, UploadBatchFile, DomainClassification
from app.services.user_service import get_or_create_user, create_default_llm_config_if_not_exists
from azure.storage.blob import ContainerClient, BlobServiceClient
from app.core.config import settings
import re
import pandas as pd
from io import BytesIO
from sqlalchemy.exc import ProgrammingError, OperationalError
from fastapi.responses import StreamingResponse
from sqlalchemy import text
import pyreadstat
import zipfile
import io
from concurrent.futures import ThreadPoolExecutor
from app.ai.langgraph_workflow.graph_executor import run_agent
from fastapi import APIRouter, WebSocket, Depends
import asyncio
from redis import Redis
from redis.commands.json.path import Path
from app.standard_query.query_processor import process_standard_query
from app.utils.token_logger import TokenLogger

import json



#Error messages variables
ERROR_MESSAGES = {
    "invalid_project_number": "Only underscores (_) are allowed as special characters.",
    "invalid_customer_name": "Hyphens (-), underscores (_), dots (.), ampersands (&), and spaces are allowed as special characters.",
    "invalid_study_number": "Only hyphens (-) and underscores (_) are allowed as special characters."
}
# Set up logging
log_file = "logs/upload.log"
os.makedirs(os.path.dirname(log_file), exist_ok=True)

logging.basicConfig(
    filename=log_file,
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)

logger = logging.getLogger(__name__)

router = APIRouter(dependencies=[Depends(azure_ad_dependency)])
ws_router = APIRouter()

@router.get(
    "/ValidateProjectNumber",
    response_model=ProjectCheckResponse,
    responses={
        200: {"description": "Project number is available"},
        409: {"description": "Project number already exists"}
    }
)
def check_project_number(
    ProjectNumber: str,
    db: Session = Depends(get_db)
):
    """
    Check if a project number is available.
    
    Parameters:
        - project_number: str - The project number to check.
    
    Returns:
        - 200 OK: If the project number is available.
        - 409 Bad Request: If the project number already exists.
    """
    existing = get_project(db, ProjectNumber=ProjectNumber)

    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "available": False,
                "message": "Project number already exists"
            }
        )
    else:
        return {
            "available": True,
            "message": "Project number is available"
        }
@router.get("/GetProjectList", response_model=List[ProjectResponse])
def list_projects(db: Session = Depends(get_db),current_user: dict = Depends(azure_ad_dependency)):
    """
    Retrieve all projects for the current user's organisation (RecordStatus == "A").
    """
    try:
        # Get or create the user with OrgId auto-assigned
        user = get_or_create_user(db, current_user)

        if not user.OrgID:
            logger.warning(f"[WARNING] User {user.UserEmail} has no OrgId assigned")
            raise HTTPException(status_code=403, detail="User is not associated with any organisation")

        # Ensure LLM config exists
        llm_config = create_default_llm_config_if_not_exists(db, user)

        project_list = []
        start_time = time.time()
        logger.debug(f"[DEBUG] Retrieving projects for OrgId={user.OrgID}")

        # Query all projects for this org
        projects = get_all_projects(db, user.OrgID)

        if not projects:
            logger.warning(f"[WARNING] No projects found for OrgId={user.OrgID}")
            return []

        for project in projects:
            created_by_username = get_username_from_user_id(project.CreatedBy, db)
            modified_by_username = get_username_from_user_id(project.ModifiedBy, db)

            project_data = {
                "ProjectNumber": project.ProjectNumber,
                "StudyNumber": project.StudyNumber,
                "CustomerName": project.CustomerName,
                "CutDate": project.CutDate,
                "ExtractionDate": project.ExtractionDate,
                "IsDatasetUploaded": project.IsDatasetUploaded,
                "CreatedBy": project.CreatedBy,
                "ModifiedBy": project.ModifiedBy,
                "ModifiedAt": project.ModifiedAt,
                "ProjectStatus": project.ProjectStatus,
                "CreatedAt": project.CreatedAt,
                "CreatedByUsername": created_by_username,
                "ModifiedByUsername": modified_by_username
            }

            project_list.append(ProjectResponse.model_validate(project_data))

        duration = time.time() - start_time
        logger.debug(f"[DEBUG] Retrieved {len(projects)} projects in {duration:.2f} seconds")

        return project_list

    except Exception as e:
        logger.error(f"[ERROR] Failed to list projects: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error retrieving projects")

@router.post("/CreateProject", response_model=ProjectCreate,status_code=201)
def create_project_with_upload(
    ProjectNumber: str = Form(...),
    StudyNumber: str = Form(...),
    CustomerName: str = Form(...),
    CutDate: Optional[date] = Form(None),
    ExtractionDate: Optional[date] = Form(None),
    
    uploaded_files: List[UploadFile] = File(default=None),
    db: Session = Depends(get_db),
    current_user: dict = Depends(azure_ad_dependency)
):
    if not re.match(r'^[a-zA-Z0-9\_]+$', ProjectNumber):
        raise HTTPException(
            status_code=422,
            detail=ERROR_MESSAGES["invalid_project_number"]
        )
    
    if not re.match(r'^[a-zA-Z0-9\-_.&\s]+$', CustomerName):
        raise HTTPException(
            status_code=422,
            detail=ERROR_MESSAGES["invalid_customer_name"]
        )

    if not re.match(r'^[a-zA-Z0-9\-_]+$', StudyNumber):
        raise HTTPException(
            status_code=422,
            detail=ERROR_MESSAGES["invalid_study_number"]
        )
    try:
        # Find the project by ProjectNumber
        user = db.query(User).filter(User.ObjectId == current_user.get("ObjectId")).first()
        project = get_project(db, ProjectNumber)
        if project:
            raise HTTPException(status_code=409, detail=f"Project with number {ProjectNumber} already exists.")
        
        if uploaded_files:
            IsDatasetUploaded = process_uploaded_file(ProjectNumber, uploaded_files, db, user.UserId)
            uploade_by=user.UserId
            uploaded_at =    datetime.now(timezone.utc)
        else:
            IsDatasetUploaded = False
            uploade_by=None
            uploaded_at = None
        
        
        # Create project in database
        project_data = ProjectCreate(
            ProjectNumber=ProjectNumber,
            StudyNumber=StudyNumber,
            CustomerName=CustomerName,
            CutDate=CutDate,
            ExtractionDate=ExtractionDate,
            UploadedBy= uploade_by,
            UploadedAt= uploaded_at,
            CreatedBy= user.UserId,
            OrgID=user.OrgID
        )
        
        db_project = create_project(db, project=project_data)
        db_project.IsDatasetUploaded = IsDatasetUploaded
        db.commit()
        db.refresh(db_project)
        
        return db_project
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    
@router.put("/EditProject", response_model=ProjectResponse,status_code=200)
def edit_project_by_number(
    ProjectNumber: str ,
    StudyNumber: str = Form(None),
    CustomerName: str = Form(None),
    CutDate: Optional[date] = Form(None),
    ExtractionDate: Optional[date] = Form(None),
    uploaded_files: List[UploadFile] = File(default=None,list=True),
    db: Session = Depends(get_db),
    current_user: dict = Depends(azure_ad_dependency)
):
    if not re.match(r'^[a-zA-Z0-9\_]+$', ProjectNumber):
        raise HTTPException(
            status_code=422,
            detail=ERROR_MESSAGES["invalid_project_number"]
        )
    
    if not re.match(r'^[a-zA-Z0-9\-_.&\s]+$', CustomerName):
        raise HTTPException(
            status_code=422,
            detail=ERROR_MESSAGES["invalid_customer_name"]
        )

    if not re.match(r'^[a-zA-Z0-9\-_]+$', StudyNumber):
        raise HTTPException(
            status_code=422,
            detail=ERROR_MESSAGES["invalid_study_number"]
        )
    try:
        # Find the project by ProjectNumber
        user = db.query(User).filter(User.ObjectId == current_user.get("ObjectId")).first()
        project = get_project_active(db, ProjectNumber)
        if not project:
            raise HTTPException(status_code=404, detail=f"Project with number {ProjectNumber} not found.")
        # Update only the fields that are provided
        if StudyNumber is not None:
            project.StudyNumber = StudyNumber
        if CustomerName is not None:
            project.CustomerName = CustomerName
        if CutDate is not None:
            project.CutDate = CutDate
        if ExtractionDate is not None:
            project.ExtractionDate = ExtractionDate
        project.ModifiedBy=user.UserId
        project.ModifiedAt=datetime.now(timezone.utc)
        

        # Handle file upload and update IsDatasetUploaded
        if uploaded_files:
            project.IsDatasetUploaded = process_uploaded_file(ProjectNumber, uploaded_files, db,user.UserId)
            project.UploadedBy=user.UserId
            project.UploadedAt=datetime.now(timezone.utc)
            

        db.commit()
        db.refresh(project)

        return ProjectResponse.model_validate(project)
        


    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    

@router.delete(
    "/DeleteProject",
    summary="Soft delete one or more projects",
    responses={
        200: {"description": "Projects soft deleted"},
        404: {"description": "Some project numbers not found"},
        409: {"description": "Some projects already deleted"}
    }
)
def soft_delete_projects(
    project_numbers: List[str] = Body(..., embed=True),
    db: Session = Depends(get_db),
    current_user: dict = Depends(azure_ad_dependency)
):
    """
    Soft delete (set RecordStatus=D) for one or more project numbers.
    """
    user = db.query(User).filter(User.ObjectId == current_user.get("ObjectId")).first()
    results = []
    not_found = []
    already_deleted = []
    deleted = []

    for proj_no in project_numbers:
        project = get_project(db, ProjectNumber=proj_no)

        if not project:
            not_found.append(proj_no)
        elif project.RecordStatus == 'D':
            already_deleted.append(proj_no)
        else:
            project.RecordStatus = 'D'
            project.DeletedAt = datetime.now(timezone.utc)
            project.DeletedBy=user.UserId
            db.add(project)
            deleted.append(proj_no)

    db.commit()

    # Raise 404 if any projects were not found
    if not_found:
        raise HTTPException(
            status_code=404,
            detail={
                "message": "project numbers were not found.",
                "not_found": not_found
            }
        )

    # Raise 409 if some were already deleted (but all others existed)
    if already_deleted:
        raise HTTPException(
            status_code=409,
            detail={
                "message": "project numbers are already deleted.",
                "already_deleted": already_deleted
            }
        )

    return {
        "message": "Projects deleted successfully",
        "deleted": deleted
    }

@router.get(
    "/GetDeletedProjectList",
    response_model=List[ProjectResponse]
)
def fetch_deleted_projects(
    db: Session = Depends(get_db),
    current_user: dict = Depends(azure_ad_dependency)
): 
        projects=get_deleted_projects(db)
        # Fetch the usernames for each project
        project_list = []
        for project in projects:
            created_by_username = get_username_from_user_id(project.CreatedBy, db)
            modified_by_username = get_username_from_user_id(project.ModifiedBy, db)
            deleted_by_username= get_username_from_user_id(project.DeletedBy, db)

            project_data = {
                "ProjectNumber": project.ProjectNumber,
                "StudyNumber": project.StudyNumber,
                "CustomerName": project.CustomerName,
                "CutDate": project.CutDate,
                "ExtractionDate": project.ExtractionDate,
                "IsDatasetUploaded": project.IsDatasetUploaded,
                "CreatedBy": project.CreatedBy,
                "ModifiedBy":project.ModifiedBy,
                "ProjectStatus": project.ProjectStatus,
                "CreatedAt": project.CreatedAt,
                "ModifiedAt":project.ModifiedAt,
                "DeletedBy":project.DeletedBy,
                "DeletedAt":project.DeletedAt,
                "CreatedByUsername": created_by_username,
                "ModifiedByUsername": modified_by_username,
                "DeleteByUsername":deleted_by_username
            }

            project_list.append(ProjectResponse.model_validate(project_data))
        
        return project_list

@router.get(
    "/GetProjectInfo",
    response_model=ProjectResponse,
    responses={
        200: {"description": "Project found"},
        404: {"description": "Project not found"}
    }
)
def get_project_by_number(
    ProjectNumber: str,
    db: Session = Depends(get_db)
):
    """
    Retrieve a project by its ProjectNumber.
    
    Parameters:
        - ProjectNumber: str - The unique identifier for the project.
    
    Returns:
        - 200 OK: If the project is found and active.
        - 404 Not Found: If the project is not found or not active.
    """
    # Get the user from the database
    project = get_project_active(db, ProjectNumber=ProjectNumber)
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project with number {ProjectNumber} not found or not active."
        )
    created_by_username = get_username_from_user_id(project.CreatedBy, db)
    modified_by_username = get_username_from_user_id(project.ModifiedBy, db)
    project_data = {
                "ProjectNumber": project.ProjectNumber,
                "StudyNumber": project.StudyNumber,                 
                "CustomerName": project.CustomerName,
                "CutDate": project.CutDate,
                "ExtractionDate": project.ExtractionDate,
                "CreatedBy": project.CreatedBy,
                "CreatedAt": project.CreatedAt,
                "ModifiedBy":project.ModifiedBy,
                "ModifiedAt":project.ModifiedAt,                
                "CreatedByUsername": created_by_username,
                "ModifiedByUsername": modified_by_username,
                "ProjectStatus": project.ProjectStatus,
            }

    
    

    return ProjectResponse.model_validate(project_data)

@router.put("/UploadFile", response_model=ProjectResponse,status_code=200)
def upload_file_to_project(
    ProjectNumber: str,
    uploaded_files: List[UploadFile] = File(...),
    db: Session = Depends(get_db),
    current_user: dict = Depends(azure_ad_dependency)
):
    """
    Upload one or more files to a specific project.
    
    Parameters:
        - project_number: str - The project number to associate the files with.
        - files: List[UploadFile] - The list of files to be uploaded.
        
    Returns:
        - 200 OK: Files uploaded successfully.
        - 404 Not Found: Project not found.
    """
    user = db.query(User).filter(User.ObjectId == current_user.get("ObjectId")).first()
    # Validate project exists
    project = get_project_active(db, ProjectNumber)
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project with number {ProjectNumber} not found."
        )

    # Process the uploaded files
    try:
        # Handle file upload and update IsDatasetUploaded
        if uploaded_files:
            project.IsDatasetUploaded = process_uploaded_file(ProjectNumber, uploaded_files, db, user.UserId)
            project.UploadedBy=user.UserId
            project.UploadedAt=datetime.now(timezone.utc)
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to upload files or empty."
            )
        db.commit()
        db.refresh(project)

        return ProjectResponse.model_validate(project)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error uploading files: {str(e)}"
        )
@router.get("/GetProjectFolderFiles")
def list_project_folder_files(
    ProjectNumber : str,
    folder: str = Query("ALL", description="Pass folder name like 'SDTM' or 'ALL' to retrieve specific/all folders"),
    db: Session = Depends(get_db)
):
    # Validate ProjectNumber is available in database
    project = get_project_active(db, ProjectNumber=ProjectNumber)
    if not project: 
        raise HTTPException(status_code=404, detail=f"Project with number {ProjectNumber} not found or not active.")
    
    # Get domain classifications for mapping
    domain_classifications = db.query(DomainClassification).all()
    domain_map = {dc.DomainName.lower(): dc.DomainFullName for dc in domain_classifications}
    
    try:
        container_client = ContainerClient.from_connection_string(
            conn_str=settings.AZURE_STORAGE_CONNECTION_STRING,
            container_name=settings.AZURE_STORAGE_CONTAINER_NAME
        )
        prefix = f"{settings.BASE_BLOB_PATH}/{ProjectNumber}/"
        blobs = container_client.list_blobs(name_starts_with=prefix)
        UploadedByUsername = get_username_from_user_id(project.UploadedBy, db) if project.UploadedBy else "Unknown"
        UploadedAt = project.UploadedAt.isoformat() if project.UploadedAt else ""
        folder_files_map = {}

        for blob in blobs:
            if not blob.name.endswith("/"):
                relative_path = blob.name.replace(prefix, "")
                parts = relative_path.split("/", 1)

                if len(parts) != 2:
                    continue  # Not inside a subfolder

                folder_name, file_full_name = parts

                if folder.upper() != "ALL" and folder.upper() != folder_name.upper():
                    continue  # Skip other folders

                file_base, file_ext = os.path.splitext(file_full_name)
                
                # Get full name from domain classification
                domain_full_name = domain_map.get(file_base.lower(), file_base)

                if folder_name not in folder_files_map:
                    folder_files_map[folder_name] = []

                file_info = {
                    "name": file_base,
                    "Fullname": domain_full_name,
                    "type": file_ext.lstrip("."),
                    "UploadedAt": UploadedAt,
                    "UploadedBy": UploadedByUsername,
                    "project_number": ProjectNumber,
                    "size_bytes": f"{round(blob.size / 1024)}kb",
                }

                folder_files_map[folder_name].append(file_info)

        if not folder_files_map:
            return {"message": f"No files found for project '{ProjectNumber}' in folder '{folder}'"}

        return folder_files_map

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error accessing Azure Blob: {str(e)}")

@router.get("/DownloadExcelFromDB")
def download_excel_from_db(
    project_number: str,
    foldername: str,
    filename: str,
    db_files: Session = Depends(get_files_db),
    db_main: Session = Depends(get_db)
):
    """
    Download data from database as Excel file with direct download link
    """
    total_start = time.time()
    logger.info(f"Starting Excel download for {project_number}/{foldername}/{filename}")

    try:
        schema = f"{project_number}_{foldername}".lower()
        table = filename.lower()

        # 1. Verify table exists
        try:
            db_files.execute(text(f"SELECT 1 FROM [{schema}].[{table}] WHERE 1=0")).fetchone()
        except Exception as e:
            raise HTTPException(status_code=404, detail=f"Table '{schema}.{table}' not found")

        # 2. Get domain full name for sheet name
        domain_classification = db_main.query(DomainClassification).filter(
            DomainClassification.DomainName.ilike(filename)
        ).first()
        sheet_name = domain_classification.DomainFullName if domain_classification else filename

        # 3. Pull column descriptions
        desc_rows = db_files.execute(
            text("""
                SELECT c.name AS ColumnName, CAST(ep.value AS NVARCHAR(4000)) AS Description
                FROM sys.columns c
                JOIN sys.tables t   ON c.object_id = t.object_id
                JOIN sys.schemas s  ON t.schema_id = s.schema_id
                LEFT JOIN sys.extended_properties ep
                    ON ep.major_id = c.object_id
                   AND ep.minor_id = c.column_id
                   AND ep.name = 'MS_Description'
                WHERE s.name = :schema AND t.name = :table
            """),
            {"schema": schema, "table": table},
        ).fetchall()
        desc_map = {r.ColumnName.upper(): (r.Description or r.ColumnName) for r in desc_rows}

        # 4. Get data from database
        query_start = time.time()
        df = pd.read_sql(
            text(f"SELECT * FROM [{schema}].[{table}]"),
            db_files.bind
        )
        logger.info(f"Data retrieved in {time.time() - query_start:.2f}s")

        # 5. Rename columns to include descriptions
        new_columns = []
        for col in df.columns:
            description = desc_map.get(col.upper(), col)
            new_columns.append(f"{description} ({col})" if description != col else col)
        df.columns = new_columns

        # 6. Create Excel file in memory
        excel_start = time.time()
        output = BytesIO()
        
        with pd.ExcelWriter(
            output,
            engine='xlsxwriter',
            engine_kwargs={'options': {'nan_inf_to_errors': True}}
        ) as writer:
            df.to_excel(writer, index=False, sheet_name=sheet_name)
        
        output.seek(0)  # Critical: reset pointer to start of stream
        logger.info(f"Excel generated in {time.time() - excel_start:.2f}s")

        # 7. Create and return download response
        total_time = time.time() - total_start
        logger.info(f"Total processing time: {total_time:.2f}s")

        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={
                "Content-Disposition": f"attachment; filename={filename}.xlsx",
                "Content-Length": str(len(output.getvalue())),
                "X-Processing-Time": f"{total_time:.2f}s"
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Download failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Excel generation failed")

@router.delete("/DeleteBlobFiles", response_model=dict)
def delete_blob_files(files: List[FileDeleteItem], db: Session = Depends(get_files_db)):
    blob_service_client = BlobServiceClient.from_connection_string(settings.AZURE_STORAGE_CONNECTION_STRING)
    container_client = blob_service_client.get_container_client(settings.AZURE_STORAGE_CONTAINER_NAME)

    deleted_files = []
    not_found_files = []
    dropped_tables = []
    failed_to_drop = []

    for file in files:
        blob_path = f"{settings.BASE_BLOB_PATH}/{file.project_number}/{file.foldername}/{file.name}.{file.type}"
        schema = f"{file.project_number}_{file.foldername}".lower()
        table = file.name.lower()

        # Step 1: Try deleting blob
        try:
            container_client.delete_blob(blob_path)
            deleted_files.append(blob_path)
        except Exception as e:
            logger.warning(f"[Blob Delete] Failed: {blob_path} - {str(e)}")
            not_found_files.append({
                "path": blob_path,
                "error": str(e),
                "error_message": f"File not found or deletion failed: {file.name.strip()}.{file.type.strip()}"
            })

        # Step 2: Try dropping the associated table
        try:
            drop_stmt = text(f"DROP TABLE [{schema}].[{table}]")
            db.execute(drop_stmt)
            db.commit()
            dropped_tables.append(f"{schema}.{table}")
        except Exception as e:
            logger.warning(f"[DB Drop] Failed: {schema}.{table} - {str(e)}")
            failed_to_drop.append({
                "table": f"{schema}.{table}",
                "error": str(e)
            })

    return {
        "deleted_blobs": deleted_files,
        "not_found_blobs": not_found_files,
        "dropped_tables": dropped_tables,
        "failed_to_drop_tables": failed_to_drop
    }

@router.get("/ViewSasDatasets")
def view_sas_datasets(
    project_number: str,
    foldername: str,
    filename: str,
    page: int = 1,
    page_size: int = 100,
    db: Session = Depends(get_files_db)
):
    try:
        schema = f"{project_number}_{foldername}".lower()
        table = filename.lower()
        offset = (page - 1) * page_size

        # Pull column descriptions
        desc_rows = db.execute(
            text("""
                SELECT c.name AS ColumnName, CAST(ep.value AS NVARCHAR(4000)) AS Description
                FROM sys.columns c
                JOIN sys.tables t   ON c.object_id = t.object_id
                JOIN sys.schemas s  ON t.schema_id = s.schema_id
                LEFT JOIN sys.extended_properties ep
                    ON ep.major_id = c.object_id
                   AND ep.minor_id = c.column_id
                   AND ep.name = 'MS_Description'
                WHERE s.name = :schema AND t.name = :table
            """),
            {"schema": schema, "table": table},
        ).fetchall()
        
        column_descriptions = {row.ColumnName: row.Description for row in desc_rows}

        query = f"""
            SELECT * FROM [{schema}].[{table}]
            ORDER BY TRY_CAST(ROWID AS INT)
            OFFSET {offset} ROWS FETCH NEXT {page_size} ROWS ONLY
        """
        df = pd.read_sql(query, db.bind)

        count_query = f"SELECT COUNT(*) as total FROM [{schema}].[{table}]"
        total = int(pd.read_sql(count_query, db.bind).iloc[0]['total'])

        return {
            "page": page,
            "page_size": page_size,
            "total": total,
            "column_descriptions": column_descriptions,
            "data": df.fillna("").to_dict(orient="records")
        }

    except (ProgrammingError, OperationalError):
        raise HTTPException(status_code=404, detail=f"Table '{schema}.{table}' does not exist.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
@router.post("/Query", tags=["AI"])
def query_langgraph(req: QueryRequest, db: Session = Depends(get_db),
    current_user: dict = Depends(azure_ad_dependency)):
    start_time = time.time()
    try:
        # Step 1: Get ProjectId from ProjectNumber
        project = db.query(Project).filter_by(ProjectNumber=req.ProjectNumber).first()
        if not project:
            return {"error": f"ProjectNumber '{req.ProjectNumber}' not found."}
        user = db.query(User).filter(User.ObjectId == current_user.get("ObjectId")).first()
        #print("User:", user.UserId, user.UserEmail)
        session = None
        if req.SessionId:
            # Step 2A: Continue existing session
            session = db.query(ClinicalQuerySession).filter_by(Id=req.SessionId).first()
            if not session:
                return {"error": f"SessionId '{req.SessionId}' not found."}
        else:
            # Step 2B: Create new session
            session = ClinicalQuerySession(
                ProjectNumber=req.ProjectNumber,
                Title=req.Question,
                IsFavorite=False,
                CreatedAt=datetime.now(timezone.utc),
                UpdatedAt=datetime.now(timezone.utc)
            )
            db.add(session)
            db.commit()
            db.refresh(session)

       # Step 3: Add user message
        # Get the next QnAGroupId (max + 1) for this session
        last_group_id = db.query(func.max(ClinicalQueryMessage.QnAGroupId))\
                        .filter(ClinicalQueryMessage.SessionId == session.Id)\
                        .scalar()

        new_group_id = (last_group_id or 0) + 1

        user_msg = ClinicalQueryMessage(
            SessionId=session.Id,
            Sender="user",
            Content=req.Question,
            Metadata={},
            CreatedAt=datetime.now(timezone.utc),
            QueryBy=user.UserId,
            ViewType=req.Type,
            QnAGroupId=new_group_id   # <-- assign new group id here
        )
        db.add(user_msg)
        db.commit()

        # Step 4: Generate LLM response
        input_tokens = 0
        output_tokens = 0
        
        if req.FlowType and req.FlowType.upper() == "STANDARD":
            answer,table_response = process_standard_query(req.ProjectNumber, req.FolderName, req.Question,req.LlmType, req.ModelName,req.STANDARD_QUERY_DATA,session.Id)
            answer_dict = json.loads(table_response)
            StandardTableContent = answer_dict
            # For standard queries, use default token counts
            input_tokens = 100
            output_tokens = 50
        else:
            # Existing AI flow
            answer = run_agent(req.ProjectNumber, req.FolderName, req.Question, req.LlmType, req.ModelName, session.Id, req.Type)
            StandardTableContent = None
            
            # Extract token usage from AI response
            try:
                answer_dict = json.loads(answer)
                token_usage = answer_dict.get('token_usage', {})
                input_tokens = token_usage.get('input_tokens', 0)
                output_tokens = token_usage.get('output_tokens', 0)
            except:
                # Fallback to default values if parsing fails
                input_tokens = 150
                output_tokens = 75
        usage = {
            "FolderName": req.FolderName,
            "ModelName": req.ModelName,
            "LLMType": req.LlmType,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }

        # Step 5: Add assistant message
        assistant_msg = ClinicalQueryMessage(
            SessionId=session.Id,
            Sender="assistant",
            Content=answer,
            Metadata=usage,
            CreatedAt=datetime.now(timezone.utc),
            QueryBy=user.UserId,
            ViewType=req.Type,
            QnAGroupId=new_group_id,   # <-- same group id as the user question
            FlowType=req.FlowType,
            StandardTableContent = StandardTableContent


        )
        db.add(assistant_msg)
        db.commit()
        db.refresh(assistant_msg)

        # Step 6: Update session timestamp
        db.query(ClinicalQuerySession).filter_by(Id=session.Id).update({
            "UpdatedAt": func.now()
        })
        db.commit()

        end_time = time.time()
        response_time = end_time - start_time
        inserted_message_id = assistant_msg.Id

        # Log token usage
        try:
            TokenLogger.log_tokens(
                db=db,
                user_id=user.UserId,
                project_number=req.ProjectNumber,
                api_endpoint="/api/Projects/Query",
                llm_provider=req.LlmType,
                llm_model=req.ModelName,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                response_time=response_time,
                session_id=session.Id,
                message_id=inserted_message_id
            )
        except Exception as e:
            logger.warning(f"Failed to log token usage: {str(e)}")

        return {
            "session_id": session.Id,
            "project_id": project.ProjectId,
            "question": req.Question,
            "answer": answer,
            "response_time_seconds": response_time,
            "Metadata":usage,
            "Id": inserted_message_id,
            "ViewType": req.Type,
            "FlowType": req.FlowType
        }

    except Exception as e:
        end_time = time.time()
        return {
            "error": str(e),
            "response_time_seconds": end_time - start_time
        }

@router.get(
    "/{ProjectNumber}/Queries",
    response_model=List[QuerySessionOut],
    tags=["AI"]
)
def list_sessions(
    ProjectNumber: str,
    db: Session = Depends(get_db),
):
    sessions = (
        db.query(ClinicalQuerySession)
          .filter_by(ProjectNumber=ProjectNumber)
          .order_by(ClinicalQuerySession.UpdatedAt.desc())
          .all()
    )

    # decorate LastMessageSnippet onto each instance
    for s in sessions:
        s.LastMessageSnippet = s.Messages[-1].Content[:100] if s.Messages else None

    return sessions


@router.get(
    "/{Sid}/Messages",
    response_model=List[MessageOut],
    tags=["AI"]
)
def get_messages(
    Sid: int,
    db: Session = Depends(get_db),
):
    msgs = (
        db.query(ClinicalQueryMessage)
        .options(joinedload(ClinicalQueryMessage.user_query_by))
        .filter_by(SessionId=Sid)
        .order_by(ClinicalQueryMessage.CreatedAt)
        .all()
    )

    # Map to response model
    response = []
    for msg in msgs:
        msg_out = MessageOut(
            Id=msg.Id,
            CreatedAt=msg.CreatedAt,
            Sender=msg.Sender,
            Content=msg.Content,
            Metadata=msg.Metadata,
            FeedbackType=msg.FeedbackType,
            FeedbackComment=msg.FeedbackComment,
            FeedbackAt=msg.FeedbackAt,
            QueryBy=msg.QueryBy,
            ViewType=msg.ViewType,
            FlowType=msg.FlowType,
            User=UserOut(
                UserId=msg.user_query_by.UserId,
                UserName=msg.user_query_by.UserName
            ) if msg.user_query_by else None
        )
        response.append(msg_out)

    return response

@router.get(
    "/LlmConfig",
    tags=["AI"],
    response_model=Dict
)
def get_user_llm_config(UserEmail: str, db: Session = Depends(get_db)):
    # 1. Get User
    user = db.query(User).filter(User.UserEmail == UserEmail).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # 2. Get active UserLLMConfig with active provider and model
    config = (
        db.query(UserLLMConfig)
        .join(LLMProvider, UserLLMConfig.ProviderId == LLMProvider.Id)
        .join(LLMModel, UserLLMConfig.ModelId == LLMModel.Id)
        .filter(
            UserLLMConfig.UserId == user.UserId,
            LLMProvider.IsActive == True,
            LLMModel.IsActive == True
        )
        .first()
    )

    if not config:
        raise HTTPException(status_code=404, detail="No active LLM config found for this user")

    # 3. Response
    return {
        "UserId": user.UserId,
        "UserEmail": user.UserEmail,
        "ProviderId": config.ProviderId,
        "ProviderName": config.Provider.Name,
        "ModelId": config.ModelId,
        "ModelName": config.Model.ModelName
    }

@router.put(
    "/LlmConfigUpdate",
    tags=["AI"],
    response_model=dict
)
def update_user_llm_config(
    config_data: UpdateLLMConfigInput,
    db: Session = Depends(get_db)
):
    # 1. Validate the model belongs to the provider and is active
    model = db.query(LLMModel).filter(
        LLMModel.Id == config_data.ModelId,
        LLMModel.ProviderId == config_data.ProviderId,
        LLMModel.IsActive == True
    ).first()

    if not model:
        raise HTTPException(status_code=400, detail="Invalid provider/model combination")

    # 2. Get existing config
    user_config = db.query(UserLLMConfig).filter_by(UserId=config_data.UserId).first()
    if not user_config:
        raise HTTPException(status_code=404, detail="User configuration not found")

    # 3. Update
    user_config.ProviderId = config_data.ProviderId
    user_config.ModelId = config_data.ModelId
    user_config.UpdatedAt = datetime.now(timezone.utc)

    db.commit()

    return {
        "message": "LLM configuration updated successfully",
        "UserId": config_data.UserId,
        "ProviderId": config_data.ProviderId,
        "ModelId": config_data.ModelId
    }

@router.get("/LlmProviders", tags=["AI"])
def get_llm_providers(db: Session = Depends(get_db)):
    providers = db.query(LLMProvider).filter(LLMProvider.IsActive == True).all()
    return [{"ProviderId": p.Id, "ProviderName": p.Name} for p in providers]

@router.get("/LlmModels", tags=["AI"])
def get_llm_models(provider_id: int, db: Session = Depends(get_db)):
    models = (
        db.query(LLMModel)
        .filter(LLMModel.ProviderId == provider_id, LLMModel.IsActive == True)
        .all()
    )
    return [{"ModelId": m.Id, "ModelName": m.ModelName} for m in models]

@router.get("/ProjectFiles")
def get_project_files(project_number: str, db: Session = Depends(get_db)):
    results = (
        db.query(
            UploadBatch.Id.label("BatchId"),
            UploadBatch.FileName,
            UploadBatch.FileSize,
            UploadBatch.UploadTime,
            UploadBatch.Status,
            User.UserName.label("UploadedBy")
        )
        .join(User, UploadBatch.UploadedBy == User.UserId)
        .filter(UploadBatch.ProjectNumber == project_number)
        .order_by(UploadBatch.UploadTime.desc())
        .all()
    )
    return [
        {
            "BatchId": row.BatchId,
            "FileName": row.FileName,
            "FileSize": row.FileSize,
            "UploadDateTime": row.UploadTime,
            "Status": row.Status,
            "UploadedBy": row.UploadedBy
        }
        for row in results
    ]

@router.get("/FileDetails")
def get_file_details(batch_id: int, db: Session = Depends(get_db)):
    # Fetch the main UploadBatch record
    batch = db.query(UploadBatch).filter(UploadBatch.Id == batch_id).first()
    if not batch:
        return {"error": "Batch not found"}

    # Get all child records
    records = (
        db.query(UploadBatchFile)
        .filter(UploadBatchFile.BatchId == batch_id)
        .order_by(UploadBatchFile.FileName)
        .all()
    )

    error_files = []
    success_count = 0

    for f in records:
        if f.ErrorNote:
            # Explicit error
            error_files.append({
                "FileName": f.FileName,
                "ErrorNote": f.ErrorNote,
                "StuckStage": None
            })
        elif f.ProcessedAt:
            # Successfully processed
            success_count += 1
        else:
            # Determine stuck stage
            if not f.StagedAt:
                stage = "Not Staged"
            elif not f.CopiedAt:
                stage = "Not Copied"
            elif not f.EnqueuedAt:
                stage = "Not Enqueued"
            elif not f.ProcessedAt:
                stage = "Not Processed"
            else:
                stage = "Unknown"

            error_files.append({
                "FileName": f.FileName,
                "ErrorNote": None,
                "StuckStage": stage
            })

    return {
        "ZipFileName": batch.FileName,
        "TotalFiles": len(records),
        "SuccessCount": success_count,
        "FailedCount": len(error_files),
        "Errors": error_files  # Includes both ErrorNote and stage-stuck issues
    }

@router.put("/UpdateFeedback", status_code=200)
def update_feedback_for_message(
    Id: int = Form(...),
    FeedbackType: Optional[str] = Form(None),          # 'Positive', 'Negative', or None
    FeedbackComment: Optional[str] = Form(None),
    db: Session = Depends(get_db),
    current_user: dict = Depends(azure_ad_dependency)
):
    try:
        # Validate FeedbackType if provided
        if FeedbackType is not None and FeedbackType not in ["Positive", "Negative"]:
            raise HTTPException(status_code=422, detail="FeedbackType must be 'Positive', 'Negative', or null")

        # Fetch the message
        message = db.query(ClinicalQueryMessage).filter(ClinicalQueryMessage.Id == Id).first()

        if not message:
            raise HTTPException(status_code=404, detail=f"ClinicalQueryMessage with Id {Id} not found.")

        # Update fields
        message.FeedbackType = FeedbackType
        message.FeedbackComment = FeedbackComment
        message.FeedbackAt = datetime.now(timezone.utc)

        db.commit()
        db.refresh(message)

        return {
            "message": "Feedback updated successfully.",
            "ClinicalQueryMessageId": message.Id,
            "FeedbackType": message.FeedbackType,
            "FeedbackComment": message.FeedbackComment,
            "FeedbackAt": message.FeedbackAt
        }
    except HTTPException:
    # re-raise so 422/404 etc. are preserved
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@ws_router.websocket("/ws/status/{project_number}")
async def websocket_status(
    websocket: WebSocket,
    project_number: str
):
    await websocket.accept()
    
    # Authenticate
    user_info = await websocket_auth(websocket)
    if not user_info:
        return
    
    logger.info(f"WebSocket connected for {user_info['UserEmail']}")
    
    # Get database session generator
    db_gen = get_websocket_db()
    db = next(db_gen)
    
    try:
        while True:
            # Use expire_all on the session instance
            db.expire_all()

            batches = db.query(UploadBatch).filter(
                UploadBatch.ProjectNumber == project_number
            ).all()

            if not batches:
                await websocket.send_json({"message": "No batches found"})
                break

            total_files = 0
            total_processed = 0
            total_failed = 0
            file_names = []

            for batch in batches:
                file_names.append(batch.FileName)
                batch_file_count = batch.FileCount or 0
                total_files += batch_file_count

                files = db.query(UploadBatchFile).filter(
                    UploadBatchFile.BatchId == batch.Id
                ).all()
                
                total_processed += sum(1 for f in files if f.Status == "Processed")
                total_failed += sum(1 for f in files if f.Status == "Error")

            total_completed = total_processed + total_failed
            status = "inprogress"
            
            if total_completed == total_files:
                status = "completed"
                await websocket.send_json({
                    "project": project_number,
                    "fileNames": file_names,
                    "total": total_files,
                    "processed": total_processed,
                    "failed": total_failed,
                    "status": status
                })
                break

            await websocket.send_json({
                "project": project_number,
                "fileNames": file_names,
                "total": total_files,
                "processed": total_processed,
                "failed": total_failed,
                "status": status
            })

            await asyncio.sleep(5)
            
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}", exc_info=True)
    finally:
        # Properly close the session
        try:
            next(db_gen)  # Trigger the finally block in the generator
        except StopIteration:
            pass
        
        if not websocket.client_state == 3:  # 3 = CLOSED
            await websocket.close()

@router.get("/redis/keys", tags=["Redis"])
def list_redis_keys(pattern: str = "*"):
    """
    List all Redis keys matching the given pattern (default is all).
    """
    try:
        r = Redis.from_url(settings.REDIS_URL)
        keys = r.keys(pattern)
        #print("keys",keys)
        decoded_keys = [key.decode() for key in keys]
        return {"keys": decoded_keys}
    except Exception as e:
        return {"error": str(e)}

@router.delete("/redis/delete-all", response_model=dict, tags=["Redis"])
def delete_all_keys():
    redis_client = Redis.from_url(settings.REDIS_URL, decode_responses=True)
    redis_client.flushdb()  # Use flushall() if you want to clear ALL databases
    return {"status": "success", "message": "All Redis keys in current DB deleted"}

@router.get("/redis/checkpoint/{thread_id}", tags=["Redis"])
def get_rejson_checkpoints(thread_id: str):
    """
    Get all ReJSON checkpoint keys and values for a given thread ID.
    """
    try:
        r = Redis.from_url(settings.REDIS_URL, decode_responses=True)
        pattern = f"checkpoint_blob:{thread_id}:*"
        keys = r.keys(pattern)
        result = {}

        for key in keys:
            try:
                value = r.json().get(key, Path.root_path())
                result[key] = value
            except Exception as e:
                result[key] = f"[Error reading ReJSON] {str(e)}"

        return result

    except Exception as e:
        return {"error": str(e)}

@router.get("/DownloadAllQueryHistory", tags=["AI"])
def download_all_query_history(UserId: int, db: Session = Depends(get_db)):
    try:
        # Get user and assistant messages in single query with join
        from sqlalchemy import and_, or_
        
        messages = db.query(ClinicalQueryMessage).filter(
            ClinicalQueryMessage.QueryBy == UserId,
            or_(
                ClinicalQueryMessage.Sender == "user",
                ClinicalQueryMessage.Sender == "assistant"
            )
        ).order_by(
            ClinicalQueryMessage.SessionId,
            ClinicalQueryMessage.QnAGroupId,
            ClinicalQueryMessage.CreatedAt
        ).all()
        
        if not messages:
            raise HTTPException(status_code=404, detail="No messages found for this user")
        
        # Group messages by SessionId and QnAGroupId
        grouped_messages = {}
        for msg in messages:
            key = (msg.SessionId, msg.QnAGroupId)
            if key not in grouped_messages:
                grouped_messages[key] = {"user": None, "assistant": None}
            grouped_messages[key][msg.Sender] = msg
        
        # Prepare data for Excel
        excel_data = []
        for (session_id, qna_group_id), msgs in grouped_messages.items():
            user_msg = msgs["user"]
            assistant_msg = msgs["assistant"]
            
            if not user_msg:
                continue
                
            ai_query = ""
            if assistant_msg and assistant_msg.Content:
                try:
                    import json
                    content = json.loads(assistant_msg.Content)
                    ai_query = content.get("query", "")
                except:
                    ai_query = ""
            
            excel_data.append({
                "SessionId": session_id,
                "QnAGroupId": qna_group_id,
                "UserQuestions": user_msg.Content,
                "AIGeneratedQuery": ai_query,
                "FeedbackType": assistant_msg.FeedbackType if assistant_msg else None,
                "FeedbackComment": assistant_msg.FeedbackComment if assistant_msg else None
            })
        
        # Create Excel file with borders
        df = pd.DataFrame(excel_data)
        output = BytesIO()
        
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='All Query History')
            
            # Add borders to all data cells
            from openpyxl.styles import Border, Side
            worksheet = writer.sheets['All Query History']
            thin_border = Border(
                left=Side(style='thin'),
                right=Side(style='thin'),
                top=Side(style='thin'),
                bottom=Side(style='thin')
            )
            
            # Apply borders to all cells with data
            for row in worksheet.iter_rows(min_row=1, max_row=len(df)+1, min_col=1, max_col=len(df.columns)):
                for cell in row:
                    cell.border = thin_border
        
        output.seek(0)
        filename = f"AllQueryHistory_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        
        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating Excel: {str(e)}")

@router.get("/bulkDownload/debug")
def debug_bulk_download(
    project_number: str,
    folder_name: str,
    db_files: Session = Depends(get_files_db)
):
    schema_name = f"{project_number}_{folder_name}"
    query = text("""
        SELECT TABLE_NAME 
        FROM INFORMATION_SCHEMA.TABLES 
        WHERE TABLE_SCHEMA = :schema_name
        AND TABLE_TYPE = 'BASE TABLE'
    """)
    
    result = db_files.execute(query, {"schema_name": schema_name})
    tables = [row[0] for row in result.fetchall()]
    
    return {"schema": schema_name, "tables": tables}

# Global progress storage (in production, use Redis or database)
progress_storage = {}

@router.get("/BulkDownload/Progress/{session_id}")
def get_download_progress(session_id: str):
    return progress_storage.get(session_id, {"status": "not_found"})

@router.get("/BulkDownload")
def bulk_download(
    project_number: str,
    folder_name: str,
    file_names: str = Query(...),
    session_id: str = Query(None),
    db_main: Session = Depends(get_db)
):
    start_time = time.time()
    logger.info(f"Starting bulk download for {project_number}_{folder_name}")
    
    processed_count = 0
    progress_data = {"processed": 0, "total": 0, "remaining": 0, "current_file": "", "status": "starting"}
    if session_id:
        progress_storage[session_id] = progress_data
    
    def process_table(table_info):
        nonlocal processed_count
        table_name, schema_name = table_info
        table_start = time.time()
        db_gen = get_files_db()
        db_files = next(db_gen)
        
        try:
            # Get column descriptions
            desc_rows = db_files.execute(
                text("""
                    SELECT c.name AS ColumnName, CAST(ep.value AS NVARCHAR(4000)) AS Description
                    FROM sys.columns c
                    JOIN sys.tables t   ON c.object_id = t.object_id
                    JOIN sys.schemas s  ON t.schema_id = s.schema_id
                    LEFT JOIN sys.extended_properties ep
                        ON ep.major_id = c.object_id
                       AND ep.minor_id = c.column_id
                       AND ep.name = 'MS_Description'
                    WHERE s.name = :schema AND t.name = :table
                """),
                {"schema": schema_name, "table": table_name},
            ).fetchall()
            desc_map = {r.ColumnName.upper(): (r.Description or r.ColumnName) for r in desc_rows}
            
            df = pd.read_sql(
                text(f"SELECT * FROM [{schema_name}].[{table_name}]"),
                db_files.bind
            )
            
            # Rename columns to include descriptions
            new_columns = []
            for col in df.columns:
                description = desc_map.get(col.upper(), col)
                new_columns.append(f"{description} ({col})" if description != col else col)
            df.columns = new_columns
            
            # Get domain full name for filename
            domain_classification = db_main.query(DomainClassification).filter(
                DomainClassification.DomainName.ilike(table_name)
            ).first()
            filename = f"{domain_classification.DomainFullName}({table_name.upper()})" if domain_classification else table_name
            
            # Create CSV file
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            file_data = csv_buffer.getvalue().encode('utf-8')
            file_size_mb = len(file_data) / (1024 * 1024)
            
            processed_count += 1
            remaining = total_count - processed_count
            
            # Update progress data
            progress_data.update({
                "processed": processed_count,
                "total": total_count,
                "remaining": remaining,
                "current_file": table_name,
                "status": "processing"
            })
            if session_id:
                progress_storage[session_id] = progress_data.copy()
            
            table_time = time.time() - table_start
            logger.info(f"Processed {table_name}: {len(df)} rows, {file_size_mb:.2f}MB in {table_time:.2f}s | Progress: {processed_count}/{total_count} (Remaining: {remaining})")
            return filename, file_data, 'csv', file_size_mb
        except Exception as e:
            logger.error(f"Error processing {table_name}: {str(e)}")
            return None, None, None, 0
        finally:
            db_files.close()
            try:
                next(db_gen)
            except StopIteration:
                pass
    
    try:
        db_gen = get_files_db()
        db_files = next(db_gen)
        
        query_start = time.time()
        
        # Handle multiple folder names (SDTM, ADaM, or both)
        folder_names = [f.strip() for f in folder_name.split(',')]
        selected_files = [f.strip() for f in file_names.split(',')]
        
        # Find tables across all specified schemas
        table_info_list = []
        for folder in folder_names:
            schema_name = f"{project_number}_{folder}"
            query = text("""
                SELECT TABLE_NAME 
                FROM INFORMATION_SCHEMA.TABLES 
                WHERE TABLE_SCHEMA = :schema_name
                AND TABLE_TYPE = 'BASE TABLE'
            """)
            
            result = db_files.execute(query, {"schema_name": schema_name})
            schema_tables = [row[0] for row in result.fetchall()]
            
            # Add matching tables with their schema info
            for table in schema_tables:
                if table in selected_files:
                    table_info_list.append((table, schema_name))
        
        total_count = len(table_info_list)
        progress_data.update({"total": total_count, "remaining": total_count, "status": "started"})
        if session_id:
            progress_storage[session_id] = progress_data.copy()
        
        try:
            next(db_gen)
        except StopIteration:
            pass
        
        query_time = time.time() - query_start
        logger.info(f"Found {total_count} tables in {query_time:.2f}s")
        
        if not table_info_list:
            raise HTTPException(status_code=404, detail=f"No tables found for the specified files")
        
        zip_start = time.time()
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_STORED) as zip_file:
            with ThreadPoolExecutor(max_workers=6) as executor:
                futures = [executor.submit(process_table, table_info) for table_info in table_info_list]
                
                total_size_mb = 0
                for future in futures:
                    filename, file_data, file_ext, file_size_mb = future.result()
                    if filename and file_data:
                        zip_file.writestr(f"{filename}.{file_ext}", file_data)
                        total_size_mb += file_size_mb
        
        zip_time = time.time() - zip_start
        logger.info(f"Created zip file in {zip_time:.2f}s")
        
        zip_buffer.seek(0)
        zip_filename = f"{project_number}_{folder_name.replace(',', '_')}.zip"
        
        total_time = time.time() - start_time
        logger.info(f"Total bulk download time: {total_time:.2f}s | Total size: {total_size_mb:.2f}MB | Files processed: {processed_count}/{total_count}")
        
        # Prepare the zip data for streaming
        zip_data = zip_buffer.read()
        
        # Only mark as completed AFTER everything is ready for download
        progress_data.update({"status": "completed"})
        if session_id:
            progress_storage[session_id] = progress_data.copy()
        
        # Create iterator for streaming
        def generate():
            yield zip_data
        
        # Create and return response
        return StreamingResponse(
            generate(),
            media_type="application/zip",
            headers={
                "Content-Disposition": f"attachment; filename={zip_filename}",
                "X-Processing-Time": f"{total_time:.2f}s",
                "X-Total-Files": str(total_count),
                "X-Processed-Files": str(processed_count),
                "X-Remaining-Files": str(total_count - processed_count),
                "X-Total-Size-MB": f"{total_size_mb:.2f}"
            }
        )
        
    except Exception as e:
        total_time = time.time() - start_time
        logger.error(f"Bulk download error after {total_time:.2f}s: {str(e)}")
        
        # Update progress with error status
        if session_id:
            progress_data.update({"status": "error", "error": str(e)})
            progress_storage[session_id] = progress_data.copy()
            
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up database connection
        try:
            db_files.close()
        except:
            pass