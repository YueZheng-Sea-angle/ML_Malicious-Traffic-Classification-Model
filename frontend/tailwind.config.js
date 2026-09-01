/** @type {import('tailwindcss').Config} */
export default {
  darkMode: "class",
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        brand: {
          50: "#eef4ff",
          100: "#d9e4ff",
          500: "#3b6cf6",
          600: "#2f57d6",
          700: "#2545ad",
        },
      },
      fontFamily: {
        sans: ["Inter", "PingFang SC", "Microsoft YaHei", "system-ui", "sans-serif"],
      },
    },
  },
  plugins: [],
};
