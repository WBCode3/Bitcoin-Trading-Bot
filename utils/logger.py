import logging
import os
from datetime import datetime

# 전역 로거 캐시
_logger_cache = {}

def setup_logger(name: str) -> logging.Logger:
    """로거 설정"""
    # 이미 설정된 로거가 있으면 반환
    if name in _logger_cache:
        return _logger_cache[name]
    
    # 로그 디렉토리 생성
    log_dir = 'logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # 로그 파일명 설정
    log_file = os.path.join(log_dir, f'{name}_{datetime.now().strftime("%Y%m%d")}.log')
    
    # 로거 생성
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # 핸들러가 이미 있는지 확인
    if not logger.handlers:
        # 파일 핸들러 설정
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # 콘솔 핸들러 설정
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # 포맷터 설정
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # 핸들러 추가
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    # 캐시에 저장
    _logger_cache[name] = logger
    
    return logger 