import DefaultTheme from 'vitepress/theme'
import type { Theme } from 'vitepress'
import BenchmarkExplorer from './components/BenchmarkExplorer.vue'
import CodecBenchmarkChart from './components/CodecBenchmarkChart.vue'
import CompressedOpsChart from './components/CompressedOpsChart.vue'
import EnergyBenchmarkChart from './components/EnergyBenchmarkChart.vue'
import KvPlayground from './components/KvPlayground.vue'
import TensorBenchmarkChart from './components/TensorBenchmarkChart.vue'
import VimgPlayground from './components/VimgPlayground.vue'
import WasmPlayground from './components/WasmPlayground.vue'
import './custom.css'

export default {
  extends: DefaultTheme,
  enhanceApp({ app }) {
    app.component('BenchmarkExplorer', BenchmarkExplorer)
    app.component('CodecBenchmarkChart', CodecBenchmarkChart)
    app.component('CompressedOpsChart', CompressedOpsChart)
    app.component('EnergyBenchmarkChart', EnergyBenchmarkChart)
    app.component('KvPlayground', KvPlayground)
    app.component('TensorBenchmarkChart', TensorBenchmarkChart)
    app.component('VimgPlayground', VimgPlayground)
    app.component('WasmPlayground', WasmPlayground)
  }
} satisfies Theme
