import React from "react";
import { FiCpu, FiHardDrive, FiServer, FiClock, FiUsers, FiActivity } from "react-icons/fi";

function SystemStatusCard({ data, isExpanded }) {
  // 데이터 구조 확인
  if (!data) return <div className="text-gray-400">데이터를 불러오는 중...</div>;

  // 리소스 사용량 상태에 따른 색상 선택
  const getResourceColor = (percent) => {
    if (percent >= 90) return "text-red-500";
    if (percent >= 75) return "text-orange-500";
    if (percent >= 50) return "text-yellow-500";
    return "text-green-500";
  };
  
  // 진행 상태 바 컴포넌트
  const ProgressBar = ({ percent, color }) => (
    <div className="w-full bg-gray-700 rounded-full h-2 mt-1 mb-1">
      <div
        className={`h-2 rounded-full ${color.replace('text-', 'bg-')}`}
        style={{ width: `${percent}%` }}
      ></div>
    </div>
  );

  return (
    <div className="h-full flex flex-col">
      {/* 시스템 리소스 상태 */}
      <div className="grid grid-cols-2 gap-4 mb-4">
        {/* CPU 상태 */}
        <div className="bg-gray-700 rounded-lg p-3">
          <div className="flex items-center mb-1">
            <FiCpu className="text-blue-400 mr-2" size={16} />
            <h3 className="text-sm font-medium text-gray-300">CPU</h3>
            <span className={`ml-auto font-semibold ${getResourceColor(data.cpu)}`}>
              {data.cpu}%
            </span>
          </div>
          <ProgressBar percent={data.cpu} color={getResourceColor(data.cpu)} />
        </div>
        
        {/* 메모리 상태 */}
        <div className="bg-gray-700 rounded-lg p-3">
          <div className="flex items-center mb-1">
            <FiServer className="text-purple-400 mr-2" size={16} />
            <h3 className="text-sm font-medium text-gray-300">메모리</h3>
            <span className={`ml-auto font-semibold ${getResourceColor(data.memory)}`}>
              {data.memory}%
            </span>
          </div>
          <ProgressBar percent={data.memory} color={getResourceColor(data.memory)} />
        </div>
        
        {/* 스토리지 상태 */}
        <div className="bg-gray-700 rounded-lg p-3">
          <div className="flex items-center mb-1">
            <FiHardDrive className="text-teal-400 mr-2" size={16} />
            <h3 className="text-sm font-medium text-gray-300">스토리지</h3>
            <span className={`ml-auto font-semibold ${getResourceColor(data.storage)}`}>
              {data.storage}%
            </span>
          </div>
          <ProgressBar percent={data.storage} color={getResourceColor(data.storage)} />
        </div>
        
        {/* 활성 연결 */}
        <div className="bg-gray-700 rounded-lg p-3">
          <div className="flex items-center mb-1">
            <FiUsers className="text-cyan-400 mr-2" size={16} />
            <h3 className="text-sm font-medium text-gray-300">활성 연결</h3>
            <span className={`ml-auto font-semibold text-gray-100`}>
              {data.activeConnections}
            </span>
          </div>
          <div className="w-full bg-gray-800 rounded-full h-2 mt-1 flex">
            {[...Array(data.activeConnections > 10 ? 10 : data.activeConnections)].map((_, i) => (
              <div 
                key={i} 
                className="h-2 w-1.5 bg-cyan-500 rounded-full mr-0.5 first:rounded-l-full last:rounded-r-full last:mr-0"
              ></div>
            ))}
          </div>
        </div>
      </div>

      {/* 시스템 세부 정보 */}
      <div className="mt-2">
        <h4 className="text-sm font-medium text-gray-400 mb-3">시스템 정보</h4>
        <div className="space-y-4">
          {/* 업타임 정보 */}
          <div className="flex">
            <div className="w-10 h-10 rounded-full bg-blue-500 bg-opacity-10 flex items-center justify-center mr-3">
              <FiActivity size={18} className="text-blue-400" />
            </div>
            <div>
              <h4 className="text-sm font-medium text-gray-300">업타임</h4>
              <p className="text-sm text-gray-400 mt-0.5">{data.uptime}</p>
            </div>
          </div>
          
          {/* 마지막 재시작 */}
          <div className="flex">
            <div className="w-10 h-10 rounded-full bg-purple-500 bg-opacity-10 flex items-center justify-center mr-3">
              <FiClock size={18} className="text-purple-400" />
            </div>
            <div>
              <h4 className="text-sm font-medium text-gray-300">마지막 재시작</h4>
              <p className="text-sm text-gray-400 mt-0.5">{data.lastRestart}</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default SystemStatusCard; 