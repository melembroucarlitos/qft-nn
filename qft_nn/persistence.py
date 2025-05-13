import os
import pathlib
from typing import Optional, Union
import torch
import boto3
from botocore.exceptions import ClientError
import json
import logging

logger = logging.getLogger(__name__)

class S3Persistence:
    def __init__(
        self,
        bucket_name: str,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        region_name: str = "us-east-1"
    ):
        """Initialize S3 persistence handler.
        
        Args:
            bucket_name: Name of the S3 bucket to use
            aws_access_key_id: AWS access key ID. If None, will try to use environment variables
            aws_secret_access_key: AWS secret access key. If None, will try to use environment variables
            region_name: AWS region name
        """
        self.bucket_name = bucket_name
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=aws_access_key_id or os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=aws_secret_access_key or os.getenv('AWS_SECRET_ACCESS_KEY'),
            region_name=region_name
        )
        
    def save_model(
        self,
        model: torch.nn.Module,
        model_name: str,
        metadata: Optional[dict] = None,
        save_dir: str = "models"
    ) -> str:
        """Save a PyTorch model to S3.
        
        Args:
            model: PyTorch model to save
            model_name: Name to give the saved model
            metadata: Optional metadata to save alongside the model
            save_dir: Directory within the S3 bucket to save the model
            
        Returns:
            S3 key where the model was saved
        """
        # Create a temporary directory to save the model
        temp_dir = pathlib.Path("temp_model_save")
        temp_dir.mkdir(exist_ok=True)
        
        try:
            # Save model state dict
            model_path = temp_dir / f"{model_name}.pt"
            torch.save(model.state_dict(), model_path)
            
            # Save metadata if provided
            if metadata:
                metadata_path = temp_dir / f"{model_name}_metadata.json"
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f)
            
            # Upload to S3
            s3_key = f"{save_dir}/{model_name}.pt"
            self.s3_client.upload_file(str(model_path), self.bucket_name, s3_key)
            
            if metadata:
                metadata_key = f"{save_dir}/{model_name}_metadata.json"
                self.s3_client.upload_file(str(metadata_path), self.bucket_name, metadata_key)
            
            return s3_key
            
        finally:
            # Clean up temporary files
            if temp_dir.exists():
                for file in temp_dir.glob("*"):
                    file.unlink()
                temp_dir.rmdir()
    
    def load_model(
        self,
        model: torch.nn.Module,
        model_name: str,
        save_dir: str = "models"
    ) -> Optional[dict]:
        """Load a PyTorch model from S3.
        
        Args:
            model: PyTorch model to load state dict into
            model_name: Name of the saved model
            save_dir: Directory within the S3 bucket where the model is saved
            
        Returns:
            Optional metadata dictionary if it exists
        """
        # Create a temporary directory to download the model
        temp_dir = pathlib.Path("temp_model_load")
        temp_dir.mkdir(exist_ok=True)
        
        try:
            # Download model state dict
            model_path = temp_dir / f"{model_name}.pt"
            s3_key = f"{save_dir}/{model_name}.pt"
            
            try:
                self.s3_client.download_file(self.bucket_name, s3_key, str(model_path))
                model.load_state_dict(torch.load(model_path))
            except ClientError as e:
                if e.response['Error']['Code'] == 'NoSuchKey':
                    logger.error(f"Model {model_name} not found in S3 bucket {self.bucket_name}")
                    return None
                raise
            
            # Try to download metadata if it exists
            metadata = None
            metadata_key = f"{save_dir}/{model_name}_metadata.json"
            metadata_path = temp_dir / f"{model_name}_metadata.json"
            
            try:
                self.s3_client.download_file(self.bucket_name, metadata_key, str(metadata_path))
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
            except ClientError:
                # Metadata doesn't exist, which is fine
                pass
            
            return metadata
            
        finally:
            # Clean up temporary files
            if temp_dir.exists():
                for file in temp_dir.glob("*"):
                    file.unlink()
                temp_dir.rmdir()
    
    def list_models(self, save_dir: str = "models") -> list[str]:
        """List all models saved in the specified directory.
        
        Args:
            save_dir: Directory within the S3 bucket to list models from
            
        Returns:
            List of model names
        """
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=f"{save_dir}/"
            )
            
            if 'Contents' not in response:
                return []
            
            # Extract model names from S3 keys
            model_names = []
            for obj in response['Contents']:
                key = obj['Key']
                if key.endswith('.pt'):
                    # Remove directory prefix and .pt extension
                    model_name = key[len(save_dir)+1:-3]
                    model_names.append(model_name)
            
            return model_names
            
        except ClientError as e:
            logger.error(f"Error listing models: {e}")
            return [] 