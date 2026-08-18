/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        brand: {
          50: '#f0f7ff',
          100: '#d8ecff',
          200: '#b8dbff',
          300: '#86c3ff',
          400: '#4ea6ff',
          500: '#1f8bff',
          600: '#0b63d6',
          700: '#083f91',
          800: '#062a66',
          900: '#041a40',
        },
      },
      boxShadow: {
        orb: '0 0 0 10px rgba(31,139,255,0.08), 0 0 40px rgba(31,139,255,0.45)',
      },
      keyframes: {
        orbPulse: {
          '0%, 100%': { transform: 'scale(1)', opacity: '1' },
          '50%': { transform: 'scale(1.06)', opacity: '0.85' },
        },
        waveform: {
          '0%, 100%': { transform: 'scaleY(0.2)' },
          '50%': { transform: 'scaleY(1)' },
        },
        softFloat: {
          '0%, 100%': { transform: 'translateY(0px)' },
          '50%': { transform: 'translateY(-2px)' },
        },
      },
      animation: {
        orbPulse: 'orbPulse 1.8s ease-in-out infinite',
        softFloat: 'softFloat 3.5s ease-in-out infinite',
        waveform: 'waveform 1.2s ease-in-out infinite',
      },
    },
  },
  plugins: [],
}

