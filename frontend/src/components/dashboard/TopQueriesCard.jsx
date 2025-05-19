import React, { useState } from "react";
import { FiSearch, FiCode, FiFileText, FiFilter } from "react-icons/fi";

function TopQueriesCard({ data, isExpanded }) {
  const [activeFilter, setActiveFilter] = useState("all");
  
  // 데이터 구조 확인
  if (!data || !Array.isArray(data) || data.length === 0) {
    return <div className="text-gray-400">데이터를 불러오는 중...</div>;
  }

  // 필터링된 데이터
  const filteredData = activeFilter === "all"
    ? data
    : data.filter(item => item.category.toLowerCase() === activeFilter.toLowerCase());

  // 카테고리별 색상 및 아이콘 매핑
  const categoryConfig = {
    SQL: {
      icon: FiCode,
      color: "bg-blue-500",
      lightColor: "bg-blue-500/20",
      textColor: "text-blue-500"
    },
    문서: {
      icon: FiFileText,
      color: "bg-emerald-500",
      lightColor: "bg-emerald-500/20",
      textColor: "text-emerald-500"
    }
  };

  return (
    <div className="h-full flex flex-col">
      {/* 필터 영역 */}
      <div className="flex space-x-2 mb-4">
        <button
          onClick={() => setActiveFilter("all")}
          className={`px-3 py-1.5 rounded-full text-xs font-medium transition-colors
            ${activeFilter === "all" 
              ? "bg-indigo-600 text-white" 
              : "bg-gray-700 text-gray-300 hover:bg-gray-600"}`}
        >
          전체
        </button>
        <button
          onClick={() => setActiveFilter("sql")}
          className={`px-3 py-1.5 rounded-full text-xs font-medium transition-colors flex items-center
            ${activeFilter === "sql" 
              ? "bg-blue-600 text-white" 
              : "bg-gray-700 text-gray-300 hover:bg-gray-600"}`}
        >
          <FiCode className="mr-1" size={12} />
          SQL
        </button>
        <button
          onClick={() => setActiveFilter("문서")}
          className={`px-3 py-1.5 rounded-full text-xs font-medium transition-colors flex items-center
            ${activeFilter === "문서" 
              ? "bg-emerald-600 text-white" 
              : "bg-gray-700 text-gray-300 hover:bg-gray-600"}`}
        >
          <FiFileText className="mr-1" size={12} />
          문서
        </button>
      </div>

      {/* 쿼리 목록 */}
      <div className={`flex-1 overflow-y-auto ${isExpanded ? 'pr-2' : ''}`}>
        <div className="space-y-2 mt-2">
          {filteredData.map((query, index) => {
            const config = categoryConfig[query.category] || {
              icon: FiSearch,
              color: "bg-gray-500",
              lightColor: "bg-gray-500/20",
              textColor: "text-gray-500"
            };
            const Icon = config.icon;
            
            return (
              <div
                key={index}
                className="flex items-center p-3 bg-gray-700/50 hover:bg-gray-700 rounded-lg transition-colors group cursor-pointer"
              >
                <div className={`w-8 h-8 rounded-full ${config.lightColor} flex items-center justify-center mr-3`}>
                  <Icon size={16} className={config.textColor} />
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center">
                    <p className="text-sm text-white truncate">{query.text}</p>
                    <span className={`ml-2 px-2 py-0.5 text-xs rounded-full ${config.lightColor} ${config.textColor}`}>
                      {query.category}
                    </span>
                  </div>
                  <p className="text-xs text-gray-400 mt-0.5">
                    {query.count}회 검색됨
                  </p>
                </div>
                <div className="opacity-0 group-hover:opacity-100 transition-opacity">
                  <button className="p-1.5 bg-gray-600 hover:bg-gray-500 rounded-lg text-gray-300">
                    <FiSearch size={14} />
                  </button>
                </div>
              </div>
            );
          })}
          
          {filteredData.length === 0 && (
            <div className="flex flex-col items-center justify-center py-8 text-gray-500">
              <FiSearch size={32} className="mb-2 opacity-50" />
              <p>검색 결과가 없습니다</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default TopQueriesCard; 