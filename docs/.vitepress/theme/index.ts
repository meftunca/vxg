import DefaultTheme from 'vitepress/theme'
import type { Theme } from 'vitepress'
import CodecBenchmarkChart from './components/CodecBenchmarkChart.vue'
import CompressedOpsChart from './components/CompressedOpsChart.vue'
import EnergyBenchmarkChart from './components/EnergyBenchmarkChart.vue'
import WasmPlayground from './components/WasmPlayground.vue'
import './custom.css'

export default {
  extends: DefaultTheme,
  enhanceApp({ app }) {
    app.component('CodecBenchmarkChart', CodecBenchmarkChart)
    app.component('CompressedOpsChart', CompressedOpsChart)
    app.component('EnergyBenchmarkChart', EnergyBenchmarkChart)
    app.component('WasmPlayground', WasmPlayground)
  }
} satisfies Theme
