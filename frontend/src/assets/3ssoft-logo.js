// 3S소프트 로고 이미지 상수
export const LOGO_IMAGE = 'https://3ssoft.co.kr/wp-content/uploads/2023/06/cropped-logo-300x104.png';

// 대체 로고 (이미지 로드 실패 시 사용할 예비 아이콘 컴포넌트용)
export const createLogoIcon = (color = '#0B3C71', size = 32) => {
  return `
    <svg xmlns="http://www.w3.org/2000/svg" width="${size}" height="${size}" viewBox="0 0 80 80" fill="none">
      <path d="M40 10L70 25L40 40L10 25L40 10Z" fill="#A5C8EF" stroke="${color}" stroke-width="1.2"/>
      <path d="M70 25L70 55L40 70L40 40L70 25Z" fill="#4280B0" stroke="${color}" stroke-width="1.2"/>
      <path d="M40 40L40 70L10 55L10 25L40 40Z" fill="${color}" stroke="${color}" stroke-width="1.2"/>
      <path d="M40 10L40 40M70 25L40 40M40 40L10 25" stroke="${color}" stroke-width="0.8" stroke-opacity="0.6"/>
    </svg>
  `;
}; 