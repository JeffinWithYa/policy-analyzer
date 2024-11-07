import os
from dotenv import load_dotenv
import uvicorn
from service.logging_config import logger

load_dotenv()

if __name__ == "__main__":
    logger.info("Starting agent service...")
    if os.getenv("MODE") != "dev":
        from service import app
        logger.info("Running in production mode")
        uvicorn.run(app, host="0.0.0.0", port=80)
    else:
        logger.info("Running in development mode")
        uvicorn.run("service:app", reload=True)
