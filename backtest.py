def _open_position(self, position_type: str, entry_price: float, current_time: datetime):
    """포지션 진입"""
    try:
        # ATR 계산
        atr = self._calculate_atr()
        
        # 포지션 크기 계산
        position_size, leverage = self.risk_manager.calculate_position_size(
            self.current_capital,
            entry_price,
            atr,
            position_type
        )
        
        # 손절가와 이익실현가 계산
        stop_loss = self.risk_manager.calculate_stop_loss(
            entry_price=entry_price,
            atr=atr,
            position_type=position_type,
            market_data=self.market_data,
            current_price=entry_price
        )
        
        take_profit = self.risk_manager.calculate_take_profit(
            entry_price=entry_price,
            atr=atr,
            position_type=position_type
        )
        
        # 포지션 정보 저장
        self.current_position = {
            'type': position_type,
            'entry_price': entry_price,
            'size': position_size,
            'leverage': leverage,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'entry_time': current_time
        }
        
        # 로깅
        self.logger.info(f"포지션 진입 - 타입: {position_type}, 진입가: {entry_price}, 크기: {position_size}, 레버리지: {leverage}")
        self.logger.info(f"손절가: {stop_loss}, 이익실현가: {take_profit}")
        
    except Exception as e:
        self.logger.error(f"포지션 진입 중 오류 발생: {e}")
        self.current_position = None 