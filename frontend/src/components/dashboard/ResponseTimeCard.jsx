import React from "react";
import { FiClock, FiTrendingDown, FiTrendingUp, FiActivity, FiCheckCircle, FiAlertCircle } from "react-icons/fi";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area } from 'recharts';

const ResponseTimeCard = ({ data, isExpanded }) => {
  // 데이터 구조 확인
  if (!data) return <div className="text-gray-400 p-4">데이터를 불러오는 중...</div>;

  // 안전하게 기본값 설정
  const safeData = {
    average: data.average ?? 0,
    trend: data.trend ?? [],
    sqlProcessing: data.sqlProcessing ?? 0.3,
    vectorSearch: data.vectorSearch ?? 0.5,
    llmGeneration: data.llmGeneration ?? 1.2
  };

  // 고정된 정적 데이터 생성 (애니메이션이 계속 동작하는 것 방지)
  const staticTrendData = [
    {date: '월', value: 2.1},
    {date: '화', value: 1.9},
    {date: '수', value: 1.7},
    {date: '목', value: 1.5},
    {date: '금', value: 1.8},
    {date: '토', value: 1.6},
    {date: '일', value: 1.5}
  ];

  // 응답 시간 트렌드 계산 (정적 데이터 기준)
  const latestValue = staticTrendData[staticTrendData.length - 1].value;
  const previousValue = staticTrendData[staticTrendData.length - 2].value;
  const trendPercent = previousValue !== 0
    ? ((latestValue - previousValue) / previousValue) * 100
    : 0;
  const isTrendDown = trendPercent <= 0;

  // 안정성 평가 (응답 시간이 짧을 수록 좋음)
  const getStabilityStatus = (avgResponseTime) => {
    if (avgResponseTime < 0.8) return { status: 'excellent', color: 'text-green-400', label: '매우 빠름', icon: FiCheckCircle };
    if (avgResponseTime < 1.5) return { status: 'good', color: 'text-blue-400', label: '양호', icon: FiActivity };
    if (avgResponseTime < 3) return { status: 'moderate', color: 'text-yellow-400', label: '보통', icon: FiClock };
    return { status: 'slow', color: 'text-red-400', label: '느림', icon: FiAlertCircle };
  };

  const stability = getStabilityStatus(safeData.average);
  const StabilityIcon = stability.icon;

  // 성능 메트릭 데이터
  const performanceMetrics = [
    { name: 'SQL 처리', value: safeData.sqlProcessing, fill: '#6366f1' },
    { name: '벡터 검색', value: safeData.vectorSearch, fill: '#3b82f6' },
    { name: 'LLM 생성', value: safeData.llmGeneration, fill: '#0ea5e9' }
  ];

  // 총 합계
  const totalProcessingTime = performanceMetrics.reduce((acc, curr) => acc + curr.value, 0);

  return (
    <div className="h-full flex flex-col">
      {/* 헤더 - 메인 지표 */}
      <div className="mb-5 flex flex-col">
        <div className="flex items-center justify-between mb-3">
          <div className="flex-1">
            <div className="flex items-center gap-2">
              <div className={`p-2 rounded-lg ${stability.status === 'excellent' ? 'bg-green-500/20' : stability.status === 'good' ? 'bg-blue-500/20' : stability.status === 'moderate' ? 'bg-yellow-500/20' : 'bg-red-500/20'}`}>
                <StabilityIcon className={stability.color} size={18} />
              </div>
              <div>
                <h3 className="text-sm font-medium text-gray-300">평균 응답 시간</h3>
                <div className="flex items-baseline mt-1">
                  <span className="text-2xl font-bold text-white">{safeData.average}</span>
                  <span className="text-gray-400 ml-1 text-xs">초</span>
                  
                  {/* 트렌드 표시 */}
                  <div className={`flex items-center ml-3 ${isTrendDown ? 'text-green-400' : 'text-red-400'}`}>
                    {isTrendDown ? (
                      <FiTrendingDown size={14} className="mr-1" />
                    ) : (
                      <FiTrendingUp size={14} className="mr-1" />
                    )}
                    <span className="text-xs font-medium">
                      {Math.abs(trendPercent).toFixed(1)}%
                    </span>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* 상태 표시 */}
          <div className={`px-2.5 py-1.5 rounded-md text-xs font-medium ${
            stability.status === 'excellent' ? 'bg-green-900/30 text-green-400 border border-green-800/40' :
            stability.status === 'good' ? 'bg-blue-900/30 text-blue-400 border border-blue-800/40' :
            stability.status === 'moderate' ? 'bg-yellow-900/30 text-yellow-400 border border-yellow-800/40' :
            'bg-red-900/30 text-red-400 border border-red-800/40'
          }`}>
            {stability.label}
          </div>
        </div>
        
        {/* 처리 시간 내역 */}
        <div className="mt-1 w-full bg-gray-800/60 rounded-full h-2.5 overflow-hidden">
          <div className="flex h-full">
            {performanceMetrics.map((metric, idx) => (
              <div 
                key={idx}
                className="h-full transition-all duration-500"
                style={{ 
                  width: `${(metric.value / totalProcessingTime) * 100}%`,
                  backgroundColor: metric.fill
                }}
              ></div>
            ))}
          </div>
        </div>
        
        {/* 범례 */}
        <div className="flex flex-wrap gap-x-4 gap-y-1 mt-2 text-xs text-gray-400">
          {performanceMetrics.map((metric, idx) => (
            <div key={idx} className="flex items-center">
              <div 
                className="w-2.5 h-2.5 rounded-full mr-1.5" 
                style={{ backgroundColor: metric.fill }}
              ></div>
              <span>{metric.name}: {metric.value}초</span>
            </div>
          ))}
        </div>
      </div>

      {/* 차트 영역 */}
      <div className="flex-1 h-36 flex flex-col">
        <div className="w-full h-full">
          <div className="h-full w-full">
            <AreaChart
              width={isExpanded ? 600 : 400}
              height={140}
              data={staticTrendData}
              margin={{ top: 5, right: 5, left: 5, bottom: 5 }}
            >
              <defs>
                <linearGradient id="responseTimeGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.6} />
                  <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" vertical={false} />
              <XAxis
                dataKey="date"
                tick={{ fill: '#9ca3af', fontSize: 10 }}
                axisLine={{ stroke: 'rgba(255,255,255,0.1)' }}
                tickLine={false}
              />
              <YAxis
                tick={{ fill: '#9ca3af', fontSize: 10 }}
                axisLine={{ stroke: 'rgba(255,255,255,0.1)' }}
                tickLine={false}
                width={30}
                tickFormatter={(value) => `${value}초`}
                domain={[1, 'auto']}
              />
              <Tooltip
                contentStyle={{ backgroundColor: 'rgba(17, 24, 39, 0.9)', borderColor: 'rgba(107, 114, 128, 0.3)', borderRadius: '0.5rem' }}
                labelStyle={{ color: '#e5e7eb', fontWeight: 'bold', marginBottom: '4px' }}
                itemStyle={{ color: '#e5e7eb', fontSize: '12px' }}
                formatter={(value) => [`${value} 초`, '응답 시간']}
                labelFormatter={(label) => `날짜: ${label}`}
                isAnimationActive={false}
              />
              <Area
                type="monotone"
                dataKey="value"
                stroke="#3b82f6"
                strokeWidth={2}
                fillOpacity={1}
                fill="url(#responseTimeGradient)"
                activeDot={{ r: 5, stroke: '#3b82f6', strokeWidth: 2, fill: '#111827' }}
                isAnimationActive={false}
              />
            </AreaChart>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ResponseTimeCard; 