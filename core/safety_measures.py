import logging
from typing import Dict, Any
from datetime import datetime, timedelta
import numpy as np

logger = logging.getLogger(__name__)

class SafetyMeasures:
    def __init__(self):
        self.last_check_time = datetime.now()
        self.price_history = []
        self.volume_history = []
        self.max_price_change = 0.0
        self.max_volume_change = 0.0
        self.emergency_exit_triggered = False
        self.system_health = {
            'last_check': datetime.now(),
            'status': 'normal',
            'issues': []
        }
        
    def check_market_shock(self, current_price: float, current_volume: float) -> bool:
        """시장 충격 감지"""
        try:
            # 가격 변화율 계산
            if self.price_history:
                price_change = abs(current_price - self.price_history[-1]) / self.price_history[-1]
                self.max_price_change = max(self.max_price_change, price_change)
                
                # 급격한 가격 변동 감지
                if price_change > 0.05:  # 5% 이상 변동
                    logger.warning(f"급격한 가격 변동 감지: {price_change:.2%}")
                    return True
                    
            # 거래량 변화율 계산
            if self.volume_history:
                volume_change = abs(current_volume - self.volume_history[-1]) / self.volume_history[-1]
                self.max_volume_change = max(self.max_volume_change, volume_change)
                
                # 급격한 거래량 변동 감지
                if volume_change > 2.0:  # 거래량 2배 이상 증가
                    logger.warning(f"급격한 거래량 변동 감지: {volume_change:.2%}")
                    return True
                    
            # 히스토리 업데이트
            self.price_history.append(current_price)
            self.volume_history.append(current_volume)
            
            # 히스토리 크기 제한
            if len(self.price_history) > 100:
                self.price_history.pop(0)
                self.volume_history.pop(0)
                
            return False
            
        except Exception as e:
            logger.error(f"시장 충격 감지 중 오류 발생: {e}")
            return False
            
    def check_system_health(self) -> Dict[str, Any]:
        """시스템 상태 점검"""
        try:
            current_time = datetime.now()
            time_since_last_check = (current_time - self.last_check_time).total_seconds()
            
            # 시스템 상태 업데이트
            self.system_health['last_check'] = current_time
            self.system_health['issues'] = []
            
            # 응답 시간 체크
            if time_since_last_check > 60:  # 1분 이상 지연
                self.system_health['status'] = 'warning'
                self.system_health['issues'].append('시스템 응답 지연')
                
            # 메모리 사용량 체크 (가정)
            memory_usage = 0.7  # 실제로는 psutil 등을 사용
            if memory_usage > 0.8:
                self.system_health['status'] = 'warning'
                self.system_health['issues'].append('높은 메모리 사용량')
                
            # CPU 사용량 체크 (가정)
            cpu_usage = 0.6  # 실제로는 psutil 등을 사용
            if cpu_usage > 0.8:
                self.system_health['status'] = 'warning'
                self.system_health['issues'].append('높은 CPU 사용량')
                
            self.last_check_time = current_time
            return self.system_health
            
        except Exception as e:
            logger.error(f"시스템 상태 점검 중 오류 발생: {e}")
            return {'status': 'error', 'issues': ['시스템 상태 점검 실패']}
            
    def should_trigger_emergency_exit(self, market_data: Dict[str, Any]) -> bool:
        """긴급 청산 필요 여부 판단"""
        try:
            if not market_data:
                return False
                
            # 현재 가격 가져오기
            current_price = market_data.get('close', 0)
            if current_price == 0:
                return False
                
            # 가격 변동성 체크
            price_change = abs(current_price - self.last_price) / self.last_price if self.last_price else 0
            if price_change > self.price_shock_threshold:
                logger.warning(f"가격 급변 감지: {price_change:.2%}")
                return True
                
            # 거래량 체크
            volume_change = abs(market_data.get('volume', 0) - self.last_volume) / self.last_volume if self.last_volume else 0
            if volume_change > self.volume_shock_threshold:
                logger.warning(f"거래량 급변 감지: {volume_change:.2%}")
                return True
                
            # 시스템 상태 체크
            if self.system_health == 'critical':
                logger.warning("시스템 상태 위험으로 긴급 청산")
                return True
                
            return False
            
        except Exception as e:
            logger.error(f"긴급 청산 판단 중 오류 발생: {e}")
            return False
            
    def get_recovery_plan(self) -> Dict[str, Any]:
        """시스템 복구 계획 수립"""
        try:
            recovery_plan = {
                'actions': [],
                'priority': 'normal'
            }
            
            # 시스템 상태에 따른 복구 계획
            if self.system_health['status'] == 'warning':
                recovery_plan['actions'].extend([
                    '시스템 리소스 최적화',
                    '불필요한 프로세스 종료',
                    '로그 정리'
                ])
                
            if self.emergency_exit_triggered:
                recovery_plan['actions'].extend([
                    '포지션 정리',
                    '리스크 파라미터 재설정',
                    '시스템 재시작'
                ])
                recovery_plan['priority'] = 'high'
                
            return recovery_plan
            
        except Exception as e:
            logger.error(f"복구 계획 수립 중 오류 발생: {e}")
            return {'actions': ['시스템 재시작'], 'priority': 'high'} 