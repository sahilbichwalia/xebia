from src.ui.app import batch_prediction,main,st
from src.config.logger import setup_logging
logger= setup_logging()

if __name__ == "__main__":
    logger.info("Application started")
    
    # Navigation
    page = st.sidebar.selectbox(
        "Navigate",
        ["Single Prediction", "Batch Prediction"]
    )
    
    logger.info(f"User selected page: {page}")
    
    if page == "Single Prediction":
        main()
    else:
        batch_prediction()
    
    logger.info("Application execution completed")
 

