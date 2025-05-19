import React, { useState } from "react";
import { FiDatabase, FiFile, FiFileText, FiFilePlus, FiCalendar, FiDownload } from "react-icons/fi";

function DataSourcesCard({ data, isExpanded }) {
  const [activeTab, setActiveTab] = useState("summary");
  
  // 데이터 구조 확인
  if (!data) return <div className="text-gray-400">데이터를 불러오는 중...</div>;

  // 파일 타입별 색상 및 아이콘 설정
  const getFileTypeConfig = (type) => {
    switch (type) {
      case "PDF":
        return { color: "bg-red-500", textColor: "text-red-500", icon: <FiFileText /> };
      case "DOCX":
        return { color: "bg-blue-500", textColor: "text-blue-500", icon: <FiFileText /> };
      case "TXT":
        return { color: "bg-gray-500", textColor: "text-gray-500", icon: <FiFile /> };
      case "PPT":
        return { color: "bg-orange-500", textColor: "text-orange-500", icon: <FiFileText /> };
      case "XLS":
        return { color: "bg-green-500", textColor: "text-green-500", icon: <FiFileText /> };
      default:
        return { color: "bg-purple-500", textColor: "text-purple-500", icon: <FiFile /> };
    }
  };

  return (
    <div className="h-full flex flex-col">
      {/* 탭 네비게이션 */}
      <div className="flex border-b border-gray-700 mb-4">
        <button
          className={`px-4 py-2 border-b-2 font-medium text-sm ${
            activeTab === "summary"
              ? "border-indigo-500 text-indigo-400"
              : "border-transparent text-gray-400 hover:text-gray-300"
          }`}
          onClick={() => setActiveTab("summary")}
        >
          요약
        </button>
        <button
          className={`px-4 py-2 border-b-2 font-medium text-sm ${
            activeTab === "recent"
              ? "border-indigo-500 text-indigo-400"
              : "border-transparent text-gray-400 hover:text-gray-300"
          }`}
          onClick={() => setActiveTab("recent")}
        >
          최근 추가된 문서
        </button>
      </div>

      {/* 요약 탭 내용 */}
      {activeTab === "summary" && (
        <div className="grid grid-cols-12 gap-6">
          {/* 요약 통계 */}
          <div className="col-span-12 md:col-span-4">
            <div className="bg-gray-800 rounded-xl p-5 h-full">
              <h3 className="text-sm font-medium text-gray-400 mb-4">데이터 소스 요약</h3>
              
              <div className="space-y-4">
                {/* 총 문서 수 */}
                <div className="flex items-center">
                  <div className="w-12 h-12 rounded-full bg-indigo-500 bg-opacity-20 flex items-center justify-center mr-4">
                    <FiDatabase className="text-indigo-400" size={20} />
                  </div>
                  <div>
                    <p className="text-sm text-gray-400">총 문서 수</p>
                    <p className="text-xl font-bold text-white">{data.totalDocuments.toLocaleString()}</p>
                  </div>
                </div>
                
                {/* 총 용량 */}
                <div className="flex items-center">
                  <div className="w-12 h-12 rounded-full bg-cyan-500 bg-opacity-20 flex items-center justify-center mr-4">
                    <FiFile className="text-cyan-400" size={20} />
                  </div>
                  <div>
                    <p className="text-sm text-gray-400">총 용량</p>
                    <p className="text-xl font-bold text-white">{data.totalSizeMB.toLocaleString()} MB</p>
                  </div>
                </div>
                
                {/* 최근 업데이트 */}
                {data.recentlyAdded && data.recentlyAdded.length > 0 && (
                  <div className="flex items-center">
                    <div className="w-12 h-12 rounded-full bg-emerald-500 bg-opacity-20 flex items-center justify-center mr-4">
                      <FiCalendar className="text-emerald-400" size={20} />
                    </div>
                    <div>
                      <p className="text-sm text-gray-400">최근 업데이트</p>
                      <p className="text-xl font-bold text-white">{data.recentlyAdded[0].date}</p>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
          
          {/* 문서 타입 분포 */}
          <div className="col-span-12 md:col-span-8">
            <div className="bg-gray-800 rounded-xl p-5 h-full">
              <h3 className="text-sm font-medium text-gray-400 mb-4">문서 타입 분포</h3>
              
              <div className="flex-1 flex flex-wrap">
                {data.types.map((item, index) => {
                  const config = getFileTypeConfig(item.type);
                  const percentage = Math.round((item.count / data.totalDocuments) * 100);
                  
                  return (
                    <div key={index} className="w-1/2 md:w-1/3 p-2">
                      <div className="bg-gray-700 rounded-lg p-3">
                        <div className="flex items-center mb-2">
                          <div className={`w-8 h-8 rounded-full ${config.color} bg-opacity-20 flex items-center justify-center mr-2`}>
                            <span className={config.textColor}>{config.icon}</span>
                          </div>
                          <div className="flex-1">
                            <div className="flex justify-between items-baseline">
                              <span className="text-sm font-medium text-white">{item.type}</span>
                              <span className="text-xs text-gray-400">{percentage}%</span>
                            </div>
                            <div className="text-xs text-gray-400">{item.count.toLocaleString()}개 파일</div>
                          </div>
                        </div>
                        <div className="w-full bg-gray-800 rounded-full h-1.5">
                          <div 
                            className={`h-1.5 rounded-full ${config.color}`}
                            style={{ width: `${percentage}%` }}
                          />
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* 최근 문서 탭 내용 */}
      {activeTab === "recent" && (
        <div className="bg-gray-800 rounded-xl p-5">
          <div className="flex justify-between items-center mb-4">
            <h3 className="text-sm font-medium text-gray-400">최근 추가된 문서</h3>
            <button className="text-xs px-3 py-1 bg-indigo-600 hover:bg-indigo-700 text-white rounded-md flex items-center transition-colors">
              <FiFilePlus size={14} className="mr-1" />
              문서 추가
            </button>
          </div>
          
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="text-left text-gray-400 text-xs border-b border-gray-700">
                  <th className="pb-2 font-medium">파일명</th>
                  <th className="pb-2 font-medium">크기</th>
                  <th className="pb-2 font-medium">추가일자</th>
                  <th className="pb-2 font-medium"></th>
                </tr>
              </thead>
              <tbody>
                {data.recentlyAdded.map((file, index) => {
                  // 파일 확장자 추출
                  const fileExt = file.name.split('.').pop().toUpperCase();
                  const config = getFileTypeConfig(fileExt);
                  
                  return (
                    <tr key={index} className="border-b border-gray-700 hover:bg-gray-700/50 transition-colors">
                      <td className="py-3 flex items-center">
                        <div className={`w-8 h-8 rounded-full ${config.color} bg-opacity-20 flex items-center justify-center mr-2`}>
                          <span className={config.textColor}>{config.icon}</span>
                        </div>
                        <span className="text-sm text-white">{file.name}</span>
                      </td>
                      <td className="py-3 text-sm text-gray-400">{file.size}</td>
                      <td className="py-3 text-sm text-gray-400">{file.date}</td>
                      <td className="py-3 text-right">
                        <button className="p-1.5 rounded-lg hover:bg-gray-700 text-gray-400 hover:text-white transition-colors">
                          <FiDownload size={16} />
                        </button>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          
          {data.recentlyAdded.length === 0 && (
            <div className="text-center py-8 text-gray-500">
              <FiFileText size={36} className="mx-auto mb-2 opacity-50" />
              <p>최근 추가된 문서가 없습니다</p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default DataSourcesCard; 