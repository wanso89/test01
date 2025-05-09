/** @type {import('tailwindcss').Config} */
export default {
  darkMode: 'class',
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        primary: {
          DEFAULT: '#3b82f6',
          50: '#eff6ff',
          100: '#dbeafe',
          200: '#bfdbfe',
          300: '#93c5fd',
          400: '#60a5fa',
          500: '#3b82f6',
          600: '#2563eb',
          700: '#1d4ed8',
          800: '#1e40af',
          900: '#1e3a8a',
          950: '#172554',
        },
        secondary: {
          DEFAULT: '#0ea5e9',
          50: '#f0f9ff',
          100: '#e0f2fe',
          200: '#bae6fd',
          300: '#7dd3fc',
          400: '#38bdf8',
          500: '#0ea5e9',
          600: '#0284c7',
          700: '#0369a1',
          800: '#075985',
          900: '#0c4a6e',
          950: '#082f49',
        },
        chat: {
          light: {
            user: '#3b82f6', // Blue-500 (변경됨)
            assistant: '#10b981', // Emerald-500 (변경됨)
            bg: '#f8fafc', // 배경색 (slate-50)
          },
          dark: {
            user: '#60a5fa', // Blue-400 (변경됨)
            assistant: '#34d399', // Emerald-400 (변경됨)
            bg: '#0f172a', // 배경색 (slate-900)
          }
        },
        zinc: {
          50: '#fafafa',
          100: '#f4f4f5',
          200: '#e4e4e7',
          300: '#d4d4d8',
          400: '#a1a1aa',
          500: '#71717a',
          600: '#52525b',
          700: '#3f3f46',
          800: '#27272a',
          900: '#18181b',
          950: '#09090b',
        },
        slate: {
          50: '#f8fafc',
          100: '#f1f5f9',
          200: '#e2e8f0',
          300: '#cbd5e1',
          400: '#94a3b8',
          500: '#64748b',
          600: '#475569',
          700: '#334155',
          800: '#1e293b',
          900: '#0f172a',
          950: '#020617',
        },
        blue: {
          50: '#eff6ff',
          100: '#dbeafe',
          200: '#bfdbfe',
          300: '#93c5fd',
          400: '#60a5fa',
          500: '#3b82f6',
          600: '#2563eb',
          700: '#1d4ed8',
          800: '#1e40af',
          900: '#1e3a8a',
          950: '#172554',
        },
        teal: {
          50: '#f0fdfa',
          100: '#ccfbf1',
          200: '#99f6e4',
          300: '#5eead4',
          400: '#2dd4bf',
          500: '#14b8a6',
          600: '#0d9488',
          700: '#0f766e',
          800: '#115e59',
          900: '#134e4a',
          950: '#042f2e',
        },
        // Open WebUI 스타일 컬러
        openwebui: {
          bg: '#0f172a',   // 배경색, 짙은 남색
          card: '#1e293b', // 카드 배경
          border: '#334155', // 경계선
          accent: '#2563eb', // 강조색(파랑)
          hover: '#334155', // 호버 효과
          text: {
            DEFAULT: '#f1f5f9', // 기본 텍스트 색상
            light: '#cbd5e1',   // 밝은 텍스트 색상
            muted: '#64748b',   // 음소거된 텍스트 색상
          }
        },
        gray: {
          750: '#2d3748', // 추가된 중간 톤
          850: '#1a202c', // 추가된 어두운 톤
          950: '#0f141a', // 거의 검정에 가까운 어두운 톤
        },
        indigo: {
          550: '#4f46e5', // 좀 더 밝은 인디고 색상
        },
        dark: {
          DEFAULT: '#121212',
          primary: '#1e1e1e',
          secondary: '#2a2a2a',
          accent: '#333333',
        },
      },
      fontFamily: {
        sans: [
          'Pretendard',
          'Noto Sans KR',
          'system-ui',
          '-apple-system',
          'BlinkMacSystemFont',
          'Segoe UI',
          'Roboto',
          'Helvetica Neue',
          'Arial',
          'sans-serif',
        ],
        mono: ['JetBrains Mono', 'Söhne Mono', 'Monaco', 'Andale Mono', 'Ubuntu Mono', 'monospace'],
      },
      animation: {
        'typing': 'typing 1s ease-in-out infinite',
        'bounce-x': 'bounce-x 1s infinite',
        'fade-in': 'fade-in 0.3s ease-out forwards',
        'slide-up': 'slide-up 0.3s ease-out forwards',
        'slide-down': 'slide-down 0.3s ease-out forwards',
        'slide-left': 'slide-left 0.3s ease-out forwards',
        'slide-right': 'slide-right 0.3s ease-out forwards',
        'spinner': 'spinner 1.5s linear infinite',
        'blob': 'blob 7s infinite',
        'progress-bar': 'progress-bar 2s linear infinite',
        'pulsate': 'pulsate 2s ease-in-out infinite',
      },
      keyframes: {
        typing: {
          '0%': { width: '0%' },
          '50%': { width: '50%' },
          '100%': { width: '100%' }
        },
        'bounce-x': {
          '0%, 100%': {
            transform: 'translateX(-25%)',
            animationTimingFunction: 'cubic-bezier(0.8, 0, 1, 1)'
          },
          '50%': {
            transform: 'translateX(0)',
            animationTimingFunction: 'cubic-bezier(0, 0, 0.2, 1)'
          }
        },
        'fade-in': {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' }
        },
        'slide-up': {
          '0%': { transform: 'translateY(20px)', opacity: '0' },
          '100%': { transform: 'translateY(0)', opacity: '1' }
        },
        'slide-down': {
          '0%': { transform: 'translateY(-20px)', opacity: '0' },
          '100%': { transform: 'translateY(0)', opacity: '1' }
        },
        'slide-left': {
          '0%': { transform: 'translateX(20px)', opacity: '0' },
          '100%': { transform: 'translateX(0)', opacity: '1' }
        },
        'slide-right': {
          '0%': { transform: 'translateX(-20px)', opacity: '0' },
          '100%': { transform: 'translateX(0)', opacity: '1' }
        },
        spinner: {
          '0%': { transform: 'rotate(0deg)' },
          '100%': { transform: 'rotate(360deg)' }
        },
        blob: {
          '0%': {
            transform: 'translate(0px, 0px) scale(1)',
          },
          '33%': {
            transform: 'translate(30px, -30px) scale(1.1)',
          },
          '66%': {
            transform: 'translate(-20px, 20px) scale(0.9)',
          },
          '100%': {
            transform: 'translate(0px, 0px) scale(1)',
          },
        },
        'progress-bar': {
          '0%': { width: '10%' },
          '50%': { width: '70%' },
          '100%': { width: '100%' },
        },
        'pulsate': {
          '0%, 100%': { opacity: 1 },
          '50%': { opacity: 0.5 },
        },
      },
      boxShadow: {
        'chat': '0 1px 3px rgba(0, 0, 0, 0.1), 0 1px 2px rgba(0, 0, 0, 0.05)',
        'chat-dark': '0 1px 3px rgba(0, 0, 0, 0.2), 0 1px 2px rgba(0, 0, 0, 0.15)',
        'message': '0 2px 5px rgba(0, 0, 0, 0.08), 0 1px 2px rgba(0, 0, 0, 0.04)',
        'message-hover': '0 4px 8px rgba(0, 0, 0, 0.1), 0 2px 4px rgba(0, 0, 0, 0.06)',
        'card': '0 1px 2px rgba(0, 0, 0, 0.1), 0 1px 3px rgba(0, 0, 0, 0.15)',
        'card-hover': '0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)',
        'subtle': '0 1px 2px 0 rgb(0 0 0 / 0.1)',
        'elevate': '0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1)',
        'inner-lg': 'inset 0 2px 4px 0 rgba(0, 0, 0, 0.15)',
        'soft-xl': '0 10px 25px -5px rgba(0, 0, 0, 0.1), 0 8px 10px -6px rgba(0, 0, 0, 0.06)',
        'soft-2xl': '0 25px 50px -12px rgba(0, 0, 0, 0.15)',
        'colored-lg': '0 10px 15px -3px rgba(var(--tw-shadow-color), 0.3), 0 4px 6px -4px rgba(var(--tw-shadow-color), 0.2)',
      },
      borderRadius: {
        'xl': '0.75rem',
        '2xl': '1rem',
        '3xl': '1.5rem',
        '4xl': '2rem',
      },
      transitionProperty: {
        'height': 'height',
        'spacing': 'margin, padding',
      },
      opacity: {
        '15': '0.15',
        '35': '0.35',
        '85': '0.85',
      },
      typography: (theme) => ({
        DEFAULT: {
          css: {
            maxWidth: '65ch',
            color: 'inherit',
            a: {
              'color': 'inherit',
              'fontWeight': '500',
              'textDecoration': 'underline',
              '&:hover': {
                opacity: '0.8',
              },
            },
            code: {
              color: theme('colors.gray.800'),
              backgroundColor: theme('colors.gray.100'),
              borderRadius: theme('borderRadius.md'),
              padding: '0.1em 0.25em',
              fontWeight: '400',
            },
            strong: {
              color: 'inherit',
            },
            'code::before': {
              content: '""',
            },
            'code::after': {
              content: '""',
            },
            pre: {
              backgroundColor: theme('colors.gray.800'),
              color: theme('colors.gray.100'),
              borderRadius: theme('borderRadius.lg'),
              border: '1px solid',
              borderColor: theme('colors.gray.700'),
              padding: '1em 1.5em',
              fontWeight: '400',
              fontSize: '0.9em',
            },
          },
        },
        dark: {
          css: {
            color: theme('colors.gray.300'),
            code: {
              color: theme('colors.gray.300'),
              backgroundColor: theme('colors.gray.700'),
            },
            pre: {
              backgroundColor: theme('colors.gray.900'),
              borderColor: theme('colors.gray.800'),
              color: theme('colors.gray.300'),
            },
            'a': {
              color: theme('colors.indigo.400'),
            },
            strong: {
              color: theme('colors.gray.100'),
            },
            h1: { color: theme('colors.gray.100') },
            h2: { color: theme('colors.gray.100') },
            h3: { color: theme('colors.gray.100') },
            h4: { color: theme('colors.gray.100') },
            blockquote: {
              color: theme('colors.gray.400'),
              borderLeftColor: theme('colors.gray.700'),
            },
          },
        },
      }),
    },
  },
  plugins: [
    function({ addComponents }) {
      addComponents({
        '.message-bubble': {
          '@apply p-4 rounded-lg shadow-message transition-all': {},
          '&:hover': {
            '@apply shadow-message-hover': {},
          },
        },
        '.user-bubble': {
          '@apply bg-blue-100 dark:bg-blue-900/30 text-slate-900 dark:text-white': {},
        },
        '.assistant-bubble': {
          '@apply bg-white dark:bg-slate-800/60 border border-slate-200 dark:border-slate-700/50 text-slate-800 dark:text-slate-100': {},
        },
      });
    },
    require('@tailwindcss/typography'),
  ],
}
