import React from "react";
import { FiClock, FiPieChart, FiCheckCircle } from "react-icons/fi";

function QueryAnalyticsCard({ data, isExpanded }) {
  // 데이터 구조 확인
  if (!data) return <div className="text-gray-400">데이터를 불러오는 중...</div>;

  // 안전하게 기본값 설정
  const safeData = {
    successRate: data.successRate ?? 0,
    averageQueryTime: data.averageQueryTime ?? 0,
    queryDistribution: data.queryDistribution ?? [],
    queryTimeDistribution: data.queryTimeDistribution ?? []
  };

  // 중요 지표 카드
  const renderMetricBox = (icon, title, value, unit, colorClass) => {
    const Icon = icon;
    return (
      <div className="bg-gray-700 rounded-xl p-4 flex flex-col">
        <div className="flex items-center mb-2">
          <div className={`w-8 h-8 rounded-full ${colorClass} bg-opacity-20 flex items-center justify-center mr-2`}>
            <Icon size={16} className={colorClass} />
          </div>
          <span className="text-gray-400 text-sm">{title}</span>
        </div>
        <div className="flex items-baseline">
          <span className="text-white text-2xl font-bold">{value}</span>
          {unit && <span className="text-gray-400 text-sm ml-1">{unit}</span>}
        </div>
      </div>
    );
  };

  return (
    <div className="h-full flex flex-col">
      {/* 상단 통계 카드 */}
      <div className="grid grid-cols-2 gap-4 mb-6">
        {renderMetricBox(
          FiCheckCircle,
          "쿼리 성공률",
          safeData.successRate,
          "%",
          "text-green-500"
        )}
        {renderMetricBox(
          FiClock,
          "평균 응답 시간",
          safeData.averageQueryTime,
          "초",
          "text-blue-500"
        )}
      </div>

      {/* 쿼리 유형 분포 */}
      <div className="flex-1 grid grid-cols-2 gap-6">
        <div className="flex flex-col">
          <h4 className="text-sm font-medium text-gray-400 mb-4">쿼리 유형 분포</h4>
          <div className="flex-1 flex items-center justify-center">
            <DonutChart 
              data={safeData.queryDistribution} 
              colors={["#6366f1", "#0ea5e9"]} 
              legend={true}
            />
          </div>
        </div>

        <div className="flex flex-col">
          <h4 className="text-sm font-medium text-gray-400 mb-4">쿼리 처리 시간 분포</h4>
          <div className="flex-1">
            <ResponseTimeBarChart data={safeData.queryTimeDistribution} />
          </div>
        </div>
      </div>
    </div>
  );
}

// 도넛 차트 컴포넌트
const DonutChart = ({ data, colors, legend = false }) => {
  // 데이터 유효성 검사
  if (!data || !Array.isArray(data) || data.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-full">
        <div className="text-gray-400 text-sm">표시할 데이터가 없습니다</div>
      </div>
    );
  }

  // 유효한 데이터만 필터링
  const validData = data.filter(item => 
    item && typeof item === 'object' && 
    item.value !== undefined && 
    !isNaN(item.value) && 
    item.category !== undefined
  );

  // 유효 데이터가 없으면 메시지 표시
  if (validData.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-full">
        <div className="text-gray-400 text-sm">유효한 데이터가 없습니다</div>
      </div>
    );
  }

  const total = validData.reduce((sum, item) => sum + item.value, 0);

  // 총합이 0이면 메시지 표시
  if (total === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-full">
        <div className="text-gray-400 text-sm">데이터 값이 모두 0입니다</div>
      </div>
    );
  }

  // SVG 속성 계산
  const size = 160;
  const strokeWidth = 30;
  const radius = (size - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;

  // 차트 데이터 계산
  let accumulatedPercent = 0;
  const chartData = validData.map((item, index) => {
    const percent = item.value / total;
    const offset = accumulatedPercent;
    accumulatedPercent += percent;
    
    return {
      ...item,
      percent,
      offset,
      color: colors[index % colors.length],
      strokeDasharray: circumference,
      strokeDashoffset: circumference * (1 - percent)
    };
  });

  return (
    <div className="flex flex-col items-center">
      <div className="relative">
        <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} className="transform -rotate-90">
          {chartData.map((item, index) => {
            // 각 아이템의 시작 위치 계산 (0도부터 시작)
            const startAngle = 2 * Math.PI * item.offset;
            const x = radius * Math.cos(startAngle);
            const y = radius * Math.sin(startAngle);
            
            return (
              <circle
                key={index}
                cx={size / 2}
                cy={size / 2}
                r={radius}
                fill="none"
                stroke={item.color}
                strokeWidth={strokeWidth}
                strokeDasharray={circumference}
                strokeDashoffset={circumference * (1 - item.percent)}
                strokeLinecap="round"
                style={{
                  transformOrigin: 'center',
                  transform: `rotate(${item.offset * 360}deg)`,
                  transition: 'stroke-dashoffset 0.5s ease'
                }}
              />
            );
          })}
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center text-center">
          <span className="text-2xl font-bold text-white">{total}</span>
          <span className="text-xs text-gray-400">총 쿼리</span>
        </div>
      </div>

      {legend && chartData.length > 0 && (
        <div className="mt-4 grid grid-cols-2 gap-x-4 gap-y-2">
          {chartData.map((item, index) => (
            <div key={index} className="flex items-center">
              <div
                className="w-3 h-3 rounded-full mr-2"
                style={{ backgroundColor: item.color }}
              ></div>
              <span className="text-xs text-gray-300">{item.category}</span>
              <span className="text-xs text-gray-400 ml-1">
                {Math.round(item.percent * 100)}%
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

// 응답 시간 분포 바 차트
const ResponseTimeBarChart = ({ data }) => {
  // 데이터 유효성 검사
  if (!data || !Array.isArray(data) || data.length === 0) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-gray-400 text-sm">표시할 데이터가 없습니다</div>
      </div>
    );
  }

  // 유효한 데이터만 필터링
  const validData = data.filter(item => 
    item && typeof item === 'object' && 
    item.value !== undefined && 
    !isNaN(item.value) && 
    item.range !== undefined
  );

  // 유효 데이터가 없으면 메시지 표시
  if (validData.length === 0) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-gray-400 text-sm">유효한 데이터가 없습니다</div>
      </div>
    );
  }

  // 최대값 찾기
  const maxValue = Math.max(...validData.map(item => item.value));

  // 모든 값이 0이면 특별 처리
  if (maxValue === 0) {
    return (
      <div className="h-full flex flex-col justify-end space-y-2">
        {validData.map((item, index) => (
          <div key={index} className="flex items-center space-x-2">
            <div className="w-16 text-right">
              <span className="text-xs text-gray-400">{item.range}</span>
            </div>
            <div className="flex-1 h-6 bg-gray-700 rounded-md overflow-hidden">
              <div className="h-full bg-gray-600 rounded-md flex items-center px-2 text-xs text-white font-medium truncate">
                0
              </div>
            </div>
          </div>
        ))}
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col justify-end space-y-2">
      {validData.map((item, index) => {
        const width = Math.max(5, (item.value / maxValue) * 100); // 최소 5% 너비 보장
        
        return (
          <div key={index} className="flex items-center space-x-2">
            <div className="w-16 text-right">
              <span className="text-xs text-gray-400">{item.range}</span>
            </div>
            <div className="flex-1 h-6 bg-gray-700 rounded-md overflow-hidden">
              <div
                className="h-full bg-gradient-to-r from-blue-600 to-indigo-500 rounded-md flex items-center px-2 text-xs text-white font-medium truncate transition-all duration-500"
                style={{ width: `${width}%` }}
              >
                {item.value}
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
};

export default QueryAnalyticsCard; 