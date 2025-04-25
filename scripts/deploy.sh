#!/bin/bash

# 로그 디렉토리 생성
mkdir -p logs

# 의존성 설치
pip install -r requirements.txt

# PM2 설치 (없는 경우)
if ! command -v pm2 &> /dev/null; then
    npm install -g pm2
fi

# PM2로 봇 실행/재시작
if pm2 list | grep -q "trading_bot"; then
    pm2 restart trading_bot
else
    pm2 start ecosystem.config.js
fi

# PM2 로그 확인
pm2 logs trading_bot 