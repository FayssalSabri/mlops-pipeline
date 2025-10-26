"""
Monitoring and logging utilities for the MLOps pipeline.
"""
import logging
import time
import psutil
import json
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path
import functools
import traceback


class PerformanceMonitor:
    """Monitor performance metrics and system resources."""
    
    def __init__(self, log_file: str = "logs/performance.log"):
        self.log_file = log_file
        self.setup_logging()
    
    def setup_logging(self):
        """Setup performance logging."""
        # Create logs directory if it doesn't exist
        Path(self.log_file).parent.mkdir(parents=True, exist_ok=True)
        
        # Setup performance logger
        self.logger = logging.getLogger("performance")
        self.logger.setLevel(logging.INFO)
        
        # File handler for performance logs
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        # Add handler if not already added
        if not self.logger.handlers:
            self.logger.addHandler(file_handler)
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system metrics."""
        try:
            return {
                "timestamp": datetime.now().isoformat(),
                "cpu_percent": psutil.cpu_percent(interval=1),
                "memory_percent": psutil.virtual_memory().percent,
                "memory_available_gb": psutil.virtual_memory().available / (1024**3),
                "disk_usage_percent": psutil.disk_usage('/').percent,
                "disk_free_gb": psutil.disk_usage('/').free / (1024**3),
                "load_average": psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None
            }
        except Exception as e:
            return {
                "timestamp": datetime.now().isoformat(),
                "error": f"Failed to get system metrics: {str(e)}"
            }
    
    def log_performance(self, operation: str, duration: float, **kwargs):
        """Log performance metrics for an operation."""
        metrics = {
            "operation": operation,
            "duration_seconds": duration,
            "timestamp": datetime.now().isoformat(),
            **kwargs
        }
        
        self.logger.info(json.dumps(metrics))
    
    def monitor_function(self, operation_name: str = None):
        """Decorator to monitor function performance."""
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                op_name = operation_name or func.__name__
                start_time = time.time()
                
                try:
                    result = func(*args, **kwargs)
                    duration = time.time() - start_time
                    self.log_performance(op_name, duration, status="success")
                    return result
                except Exception as e:
                    duration = time.time() - start_time
                    self.log_performance(
                        op_name, 
                        duration, 
                        status="error",
                        error=str(e),
                        traceback=traceback.format_exc()
                    )
                    raise
            
            return wrapper
        return decorator


class ModelMonitor:
    """Monitor model performance and predictions."""
    
    def __init__(self, log_file: str = "logs/model_monitor.log"):
        self.log_file = log_file
        self.setup_logging()
    
    def setup_logging(self):
        """Setup model monitoring logging."""
        # Create logs directory if it doesn't exist
        Path(self.log_file).parent.mkdir(parents=True, exist_ok=True)
        
        # Setup model logger
        self.logger = logging.getLogger("model_monitor")
        self.logger.setLevel(logging.INFO)
        
        # File handler for model logs
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        # Add handler if not already added
        if not self.logger.handlers:
            self.logger.addHandler(file_handler)
    
    def log_prediction(self, input_data: Dict, prediction: Any, 
                      probability: Optional[float] = None, 
                      model_type: str = "unknown",
                      processing_time: Optional[float] = None):
        """Log prediction details."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": "prediction",
            "model_type": model_type,
            "input_data": input_data,
            "prediction": prediction,
            "probability": probability,
            "processing_time_seconds": processing_time
        }
        
        self.logger.info(json.dumps(log_entry))
    
    def log_model_load(self, model_path: str, model_type: str, 
                      load_time: float, success: bool = True):
        """Log model loading events."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": "model_load",
            "model_path": model_path,
            "model_type": model_type,
            "load_time_seconds": load_time,
            "success": success
        }
        
        self.logger.info(json.dumps(log_entry))
    
    def log_model_training(self, model_type: str, training_time: float,
                          train_accuracy: float, test_accuracy: float,
                          test_auc: float, n_samples: int):
        """Log model training events."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": "model_training",
            "model_type": model_type,
            "training_time_seconds": training_time,
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy,
            "test_auc": test_auc,
            "n_samples": n_samples
        }
        
        self.logger.info(json.dumps(log_entry))
    
    def log_error(self, error_type: str, error_message: str, 
                 context: Optional[Dict] = None):
        """Log errors and exceptions."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": "error",
            "error_type": error_type,
            "error_message": error_message,
            "context": context or {},
            "traceback": traceback.format_exc()
        }
        
        self.logger.error(json.dumps(log_entry))


class HealthChecker:
    """Check system and model health."""
    
    def __init__(self, model_predictor=None):
        self.model_predictor = model_predictor
        self.performance_monitor = PerformanceMonitor()
        self.model_monitor = ModelMonitor()
    
    def check_system_health(self) -> Dict[str, Any]:
        """Check overall system health."""
        health_status = {
            "timestamp": datetime.now().isoformat(),
            "overall_status": "healthy",
            "checks": {}
        }
        
        # Check system resources
        system_metrics = self.performance_monitor.get_system_metrics()
        
        # CPU check
        cpu_percent = system_metrics.get("cpu_percent", 0)
        if cpu_percent > 90:
            health_status["checks"]["cpu"] = {
                "status": "warning",
                "message": f"High CPU usage: {cpu_percent}%"
            }
        else:
            health_status["checks"]["cpu"] = {
                "status": "healthy",
                "message": f"CPU usage: {cpu_percent}%"
            }
        
        # Memory check
        memory_percent = system_metrics.get("memory_percent", 0)
        if memory_percent > 90:
            health_status["checks"]["memory"] = {
                "status": "warning",
                "message": f"High memory usage: {memory_percent}%"
            }
        else:
            health_status["checks"]["memory"] = {
                "status": "healthy",
                "message": f"Memory usage: {memory_percent}%"
            }
        
        # Disk check
        disk_percent = system_metrics.get("disk_usage_percent", 0)
        if disk_percent > 90:
            health_status["checks"]["disk"] = {
                "status": "warning",
                "message": f"High disk usage: {disk_percent}%"
            }
        else:
            health_status["checks"]["disk"] = {
                "status": "healthy",
                "message": f"Disk usage: {disk_percent}%"
            }
        
        # Check if any checks failed
        if any(check["status"] == "warning" for check in health_status["checks"].values()):
            health_status["overall_status"] = "warning"
        
        return health_status
    
    def check_model_health(self) -> Dict[str, Any]:
        """Check model health."""
        health_status = {
            "timestamp": datetime.now().isoformat(),
            "overall_status": "healthy",
            "checks": {}
        }
        
        if self.model_predictor is None:
            health_status["checks"]["model"] = {
                "status": "error",
                "message": "Model predictor not available"
            }
            health_status["overall_status"] = "error"
            return health_status
        
        try:
            # Test model availability
            model_info = self.model_predictor.get_model_info()
            health_status["checks"]["model"] = {
                "status": "healthy",
                "message": f"Model loaded: {model_info.get('model_type', 'unknown')}"
            }
            
            # Test prediction capability
            test_data = {
                "feature1": 45.5,
                "feature2": 28.3,
                "feature3": 22.1
            }
            
            start_time = time.time()
            prediction = self.model_predictor.predict(test_data)
            prediction_time = time.time() - start_time
            
            health_status["checks"]["prediction"] = {
                "status": "healthy",
                "message": f"Prediction working, time: {prediction_time:.3f}s"
            }
            
        except Exception as e:
            health_status["checks"]["model"] = {
                "status": "error",
                "message": f"Model error: {str(e)}"
            }
            health_status["overall_status"] = "error"
        
        return health_status
    
    def get_comprehensive_health(self) -> Dict[str, Any]:
        """Get comprehensive health status."""
        system_health = self.check_system_health()
        model_health = self.check_model_health()
        
        overall_status = "healthy"
        if (system_health["overall_status"] == "warning" or 
            model_health["overall_status"] == "error"):
            overall_status = "warning"
        elif (system_health["overall_status"] == "error" or 
              model_health["overall_status"] == "error"):
            overall_status = "error"
        
        return {
            "timestamp": datetime.now().isoformat(),
            "overall_status": overall_status,
            "system_health": system_health,
            "model_health": model_health
        }


def setup_logging(log_level: str = "INFO", log_file: str = "logs/app.log"):
    """Setup application-wide logging."""
    # Create logs directory if it doesn't exist
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    # Set specific loggers
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("fastapi").setLevel(logging.INFO)


# Global monitoring instances
performance_monitor = PerformanceMonitor()
model_monitor = ModelMonitor()

# Convenience functions
def monitor_performance(operation_name: str = None):
    """Decorator to monitor function performance."""
    return performance_monitor.monitor_function(operation_name)

def log_prediction(input_data: Dict, prediction: Any, **kwargs):
    """Log prediction details."""
    model_monitor.log_prediction(input_data, prediction, **kwargs)

def log_error(error_type: str, error_message: str, **kwargs):
    """Log errors."""
    model_monitor.log_error(error_type, error_message, **kwargs)
