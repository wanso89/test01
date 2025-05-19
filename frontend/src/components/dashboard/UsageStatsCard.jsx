import React from "react";
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell, Legend } from 'recharts';
import { FiUsers, FiMessageCircle, FiSearch, FiTrendingUp, FiBarChart2 } from "react-icons/fi";

const UsageStatsCard = ({ data, isExpanded }) => {
  // 데이터 구조 확인
  if (!data) return <div className="text-gray-400 p-4">데이터를 불러오는 중...</div>;

  // 안전하게 기본값 설정
  const safeData = {
    totalQueries: data.totalQueries ?? 0,
    totalChats: data.totalChats ?? 0,
    activeUsers: data.activeUsers ?? 0,
    averageQueriesPerDay: data.averageQueriesPerDay ?? 0,
    queryCountByDate: data.queryCountByDate ?? []
  };

  // 추가 통계 계산
  const last30DaysQueries = safeData.queryCountByDate.slice(-30).reduce((sum, item) => sum + item.value, 0);
  const last7DaysQueries = safeData.queryCountByDate.slice(-7).reduce((sum, item) => sum + item.value, 0);
  const averageLast7Days = last7DaysQueries / 7;

  // 사용량 추세 계산
  const trend = safeData.queryCountByDate.length > 2 ? 
    (safeData.queryCountByDate[safeData.queryCountByDate.length - 1].value - 
     safeData.queryCountByDate[safeData.queryCountByDate.length - 2].value) /
     safeData.queryCountByDate[safeData.queryCountByDate.length - 2].value * 100 : 0;
  
  // 타입별 쿼리 분포 데이터 (예시 데이터, 실제로는 API에서 받아와야 함)
  const queryTypeData = [
    { name: 'SQL 질의', value: Math.round(safeData.totalQueries * 0.58) },
    { name: '문서 검색', value: Math.round(safeData.totalQueries * 0.42) }
  ];
  
  // 색상 설정
  const COLORS = ['#6366f1', '#10b981', '#f59e0b', '#ef4444'];
  
  // 차트 데이터 생성
  const chartData = safeData.queryCountByDate.map(item => ({
    ...item,
    showValue: item.value
  }));

  return (
    <div className="h-full flex flex-col">
      {/* 주요 지표 섹션 */}
      <div className="mb-5">
        <div className="grid grid-cols-2 gap-4">
          {/* 총 쿼리 수 */}
          <div className="bg-gradient-to-br from-blue-900/30 to-indigo-900/30 rounded-xl p-4 border border-blue-800/20">
            <div className="flex items-center mb-1">
              <div className="w-8 h-8 rounded-full bg-blue-500/20 flex items-center justify-center mr-2">
                <FiSearch size={16} className="text-blue-400" />
              </div>
              <span className="text-gray-400 text-sm">총 쿼리 수</span>
            </div>
            <div className="flex items-baseline">
              <span className="text-white text-2xl font-bold">{safeData.totalQueries.toLocaleString()}</span>
              {trend !== 0 && (
                <div className={`flex items-center ml-3 text-xs ${trend > 0 ? 'text-green-400' : 'text-red-400'}`}>
                  <FiTrendingUp size={14} className="mr-1" />
                  <span>{Math.abs(trend).toFixed(1)}%</span>
                </div>
              )}
            </div>
          </div>
          
          {/* 활성 사용자 */}
          <div className="bg-gradient-to-br from-emerald-900/30 to-teal-900/30 rounded-xl p-4 border border-emerald-800/20">
            <div className="flex items-center mb-1">
              <div className="w-8 h-8 rounded-full bg-emerald-500/20 flex items-center justify-center mr-2">
                <FiUsers size={16} className="text-emerald-400" />
              </div>
              <span className="text-gray-400 text-sm">활성 사용자</span>
            </div>
            <div className="text-white text-2xl font-bold">
              {safeData.activeUsers.toLocaleString()}
            </div>
          </div>
        </div>
      </div>
      
      {/* 차트 섹션 */}
      <div className="flex-1 flex flex-col">
        <div className="mb-2 flex items-center justify-between">
          <h3 className="text-sm font-medium text-gray-400">쿼리 추세</h3>
          <div className="flex items-center text-xs text-gray-400">
            <span className="mr-2">일평균:</span>
            <span className="bg-gray-800 px-2 py-0.5 rounded font-medium text-blue-400">
              {safeData.averageQueriesPerDay.toFixed(1)}
            </span>
          </div>
        </div>
        
        {/* 추세 차트 */}
        <div className="flex-1 min-h-[200px] w-full">
          <ResponsiveContainer width="100%" height={isExpanded ? 400 : "100%"}>
            <AreaChart
              data={chartData}
              margin={{ top: 5, right: 5, left: 5, bottom: 5 }}
            >
              <defs>
                <linearGradient id="colorQueries" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#6366f1" stopOpacity={0.8} />
                  <stop offset="95%" stopColor="#6366f1" stopOpacity={0} />
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
              />
              <Tooltip
                contentStyle={{ backgroundColor: 'rgba(17, 24, 39, 0.9)', borderColor: 'rgba(107, 114, 128, 0.3)', borderRadius: '0.5rem' }}
                labelStyle={{ color: '#e5e7eb', fontWeight: 'bold', marginBottom: '4px' }}
                itemStyle={{ color: '#e5e7eb', fontSize: '12px' }}
                formatter={(value) => [`${value}`, '쿼리 수']}
                labelFormatter={(label) => `날짜: ${label}`}
              />
              <Area
                type="monotone"
                dataKey="value"
                name="쿼리 수"
                stroke="#6366f1"
                strokeWidth={2}
                fillOpacity={1}
                fill="url(#colorQueries)"
                activeDot={{ r: 5, stroke: '#6366f1', strokeWidth: 2, fill: '#111827' }}
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
        
        {/* 하단 통계 */}
        {isExpanded && (
          <div className="mt-6 grid grid-cols-2 gap-6">
            {/* 타입별 쿼리 분포 */}
            <div className="bg-gray-800/40 rounded-xl p-4 border border-gray-700/30">
              <h4 className="text-sm font-medium text-gray-400 mb-3">쿼리 유형 분포</h4>
              <div className="h-[200px] flex items-center justify-center">
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={queryTypeData}
                      innerRadius={60}
                      outerRadius={80}
                      paddingAngle={5}
                      dataKey="value"
                      nameKey="name"
                    >
                      {queryTypeData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                      ))}
                    </Pie>
                    <Tooltip
                      contentStyle={{ backgroundColor: 'rgba(17, 24, 39, 0.9)', borderColor: 'rgba(107, 114, 128, 0.3)', borderRadius: '0.5rem' }}
                      labelStyle={{ color: '#e5e7eb', fontWeight: 'bold', marginBottom: '4px' }}
                      itemStyle={{ color: '#e5e7eb', fontSize: '12px' }}
                    />
                    <Legend 
                      verticalAlign="bottom" 
                      height={36} 
                      formatter={(value, entry, index) => (
                        <span className="text-gray-300 text-xs">{value}</span>
                      )}
                    />
                  </PieChart>
                </ResponsiveContainer>
              </div>
            </div>
            
            {/* 추가 통계 */}
            <div className="bg-gray-800/40 rounded-xl p-4 border border-gray-700/30">
              <h4 className="text-sm font-medium text-gray-400 mb-4">통계 요약</h4>
              <div className="space-y-4">
                <div>
                  <div className="flex justify-between items-baseline mb-1">
                    <span className="text-xs text-gray-400">최근 7일 쿼리</span>
                    <span className="text-sm text-white font-medium">{last7DaysQueries.toLocaleString()}</span>
                  </div>
                  <div className="w-full bg-gray-700/50 rounded-full h-1.5">
                    <div 
                      className="bg-blue-500 h-1.5 rounded-full" 
                      style={{ width: `${Math.min(100, (last7DaysQueries / last30DaysQueries) * 100)}%` }}
                    ></div>
                  </div>
                </div>
                
                <div>
                  <div className="flex justify-between items-baseline mb-1">
                    <span className="text-xs text-gray-400">최근 30일 쿼리</span>
                    <span className="text-sm text-white font-medium">{last30DaysQueries.toLocaleString()}</span>
                  </div>
                  <div className="w-full bg-gray-700/50 rounded-full h-1.5">
                    <div 
                      className="bg-indigo-500 h-1.5 rounded-full" 
                      style={{ width: `${Math.min(100, (last30DaysQueries / safeData.totalQueries) * 100)}%` }}
                    ></div>
                  </div>
                </div>
                
                <div>
                  <div className="flex justify-between items-baseline mb-1">
                    <span className="text-xs text-gray-400">쿼리/대화 비율</span>
                    <span className="text-sm text-white font-medium">
                      {safeData.totalChats > 0 
                        ? (safeData.totalQueries / safeData.totalChats).toFixed(1) 
                        : 'N/A'}
                    </span>
                  </div>
                  <div className="w-full bg-gray-700/50 rounded-full h-1.5">
                    <div 
                      className="bg-emerald-500 h-1.5 rounded-full" 
                      style={{ 
                        width: `${Math.min(100, safeData.totalChats > 0 
                          ? (safeData.totalQueries / safeData.totalChats / 5) * 100 
                          : 0)}%` 
                      }}
                    ></div>
                  </div>
                </div>
                
                <div className="mt-5 grid grid-cols-2 gap-3">
                  <div className="bg-gray-700/30 rounded-lg p-3">
                    <div className="text-xs text-gray-400 mb-1">일일 평균 (7일)</div>
                    <div className="text-lg font-medium text-white">
                      {averageLast7Days.toFixed(1)}
                    </div>
                  </div>
                  <div className="bg-gray-700/30 rounded-lg p-3">
                    <div className="text-xs text-gray-400 mb-1">총 대화 수</div>
                    <div className="text-lg font-medium text-white">
                      {safeData.totalChats.toLocaleString()}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default UsageStatsCard; 