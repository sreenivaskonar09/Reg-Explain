import React, { useState, useEffect, useCallback } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from './components/ui/card';
import { Button } from './components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './components/ui/tabs';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './components/ui/select';
import { Slider } from './components/ui/slider';
import { Badge } from './components/ui/badge';
import { Progress } from './components/ui/progress';
import { Alert, AlertDescription, AlertTitle } from './components/ui/alert';
import { ScrollArea } from './components/ui/scroll-area';
import { Separator } from './components/ui/separator';
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, 
  ResponsiveContainer, BarChart, Bar, AreaChart, Area, ComposedChart
} from 'recharts';
import { 
  TrendingDown, AlertTriangle, CheckCircle, Activity, Database, 
  Brain, FileText, Download, RefreshCw, Play, ChevronRight,
  Building2, DollarSign, Percent, Shield, Zap
} from 'lucide-react';
import axios from 'axios';
import './App.css';

const API_URL = process.env.REACT_APP_BACKEND_URL;

function App() {
  // State management
  const [scenarios, setScenarios] = useState({});
  const [selectedScenario, setSelectedScenario] = useState('baseline');
  const [bankData, setBankData] = useState([]);
  const [forecastResults, setForecastResults] = useState(null);
  const [modelStatus, setModelStatus] = useState(null);
  const [loading, setLoading] = useState({});
  const [trainingProgress, setTrainingProgress] = useState(0);
  const [currentConditions, setCurrentConditions] = useState(null);
  const [shapExplanation, setShapExplanation] = useState(null);
  const [aiExplanation, setAiExplanation] = useState(null);
  
  // Training configuration
  const [trainingConfig, setTrainingConfig] = useState({
    lstm_units: 64,
    lstm_dropout: 0.2,
    lstm_epochs: 100,
    xgb_estimators: 200,
    xgb_depth: 6,
    xgb_learning_rate: 0.05
  });

  // Custom scenario parameters
  const [customParams, setCustomParams] = useState({
    unemployment: 10.0,
    vix: 72,
    hpi_decline: -25,
    rate_shock: 300
  });

  // Fetch initial data
  useEffect(() => {
    fetchScenarios();
    fetchModelStatus();
    fetchCurrentMacro();
  }, []);

  const fetchScenarios = async () => {
    try {
      const response = await axios.get(`${API_URL}/api/scenarios`);
      setScenarios(response.data);
    } catch (error) {
      console.error('Error fetching scenarios:', error);
    }
  };

  const fetchModelStatus = async () => {
    try {
      const response = await axios.get(`${API_URL}/api/model/status`);
      setModelStatus(response.data);
    } catch (error) {
      console.error('Error fetching model status:', error);
    }
  };

  const fetchCurrentMacro = async () => {
    try {
      const response = await axios.get(`${API_URL}/api/macro/current`);
      setCurrentConditions(response.data);
    } catch (error) {
      console.error('Error fetching macro data:', error);
    }
  };

  const generateSyntheticBanks = async () => {
    setLoading(prev => ({ ...prev, banks: true }));
    try {
      const response = await axios.post(`${API_URL}/api/banks/synthetic?n_banks=10&n_quarters=36`);
      const banksResponse = await axios.get(`${API_URL}/api/banks`);
      setBankData(banksResponse.data);
      alert(`Generated ${response.data.total_records} bank records`);
    } catch (error) {
      console.error('Error generating banks:', error);
    }
    setLoading(prev => ({ ...prev, banks: false }));
  };

  const trainModels = async () => {
    setLoading(prev => ({ ...prev, training: true }));
    setTrainingProgress(10);
    
    try {
      // Simulate progress
      const progressInterval = setInterval(() => {
        setTrainingProgress(prev => Math.min(prev + 10, 90));
      }, 2000);

      const response = await axios.post(`${API_URL}/api/model/train`, trainingConfig);
      
      clearInterval(progressInterval);
      setTrainingProgress(100);
      setModelStatus(prev => ({ ...prev, all_ready: true, ...response.data }));
      
      setTimeout(() => setTrainingProgress(0), 1000);
    } catch (error) {
      console.error('Error training models:', error);
    }
    setLoading(prev => ({ ...prev, training: false }));
  };

  const runForecast = async () => {
    setLoading(prev => ({ ...prev, forecast: true }));
    try {
      const response = await axios.post(`${API_URL}/api/forecast/run`, {
        scenario_type: selectedScenario,
        n_quarters: 9
      });
      
      // Fetch full results
      const fullResults = await axios.get(`${API_URL}/api/forecast/results/${response.data.result_id}`);
      setForecastResults(fullResults.data);
    } catch (error) {
      console.error('Error running forecast:', error);
    }
    setLoading(prev => ({ ...prev, forecast: false }));
  };

  const fetchShapExplanation = async () => {
    setLoading(prev => ({ ...prev, shap: true }));
    try {
      const response = await axios.post(`${API_URL}/api/explain/global?scenario_type=${selectedScenario}`);
      setShapExplanation(response.data);
    } catch (error) {
      console.error('Error fetching SHAP:', error);
    }
    setLoading(prev => ({ ...prev, shap: false }));
  };

  const fetchAiExplanation = async () => {
    setLoading(prev => ({ ...prev, ai: true }));
    try {
      const response = await axios.post(`${API_URL}/api/explain/ai?scenario_type=${selectedScenario}`);
      setAiExplanation(response.data.ai_explanation);
    } catch (error) {
      console.error('Error fetching AI explanation:', error);
    }
    setLoading(prev => ({ ...prev, ai: false }));
  };

  // Calculate metrics from forecast
  const getMetrics = useCallback(() => {
    if (!forecastResults?.trajectory) return null;
    
    const trajectory = forecastResults.trajectory;
    const minCet1 = Math.min(...trajectory.map(t => t.cet1_ratio));
    const initialCet1 = trajectory.filter(t => t.quarter === 1)[0]?.cet1_ratio || 0.12;
    const breachCount = new Set(trajectory.filter(t => t.breach).map(t => t.bank_id)).size;
    const totalBanks = new Set(trajectory.map(t => t.bank_id)).size;
    const totalLosses = trajectory.reduce((sum, t) => sum + t.credit_losses, 0);
    
    return {
      minCet1: minCet1 * 100,
      decline: (initialCet1 - minCet1) * 100,
      breachRate: (breachCount / totalBanks) * 100,
      breachCount,
      totalBanks,
      totalLosses: totalLosses / 1e9
    };
  }, [forecastResults]);

  const metrics = getMetrics();

  // Prepare chart data
  const getTrajectoryData = useCallback(() => {
    if (!forecastResults?.trajectory) return [];
    
    const byQuarter = {};
    forecastResults.trajectory.forEach(t => {
      if (!byQuarter[t.quarter]) {
        byQuarter[t.quarter] = { quarter: `Q${t.quarter}`, values: [] };
      }
      byQuarter[t.quarter].values.push(t.cet1_ratio * 100);
    });
    
    return Object.values(byQuarter).map(q => ({
      quarter: q.quarter,
      avgCet1: q.values.reduce((a, b) => a + b, 0) / q.values.length,
      minCet1: Math.min(...q.values),
      maxCet1: Math.max(...q.values)
    }));
  }, [forecastResults]);

  const getScenarioData = useCallback(() => {
    if (!forecastResults?.scenario_data) return [];
    return forecastResults.scenario_data.map(s => ({
      quarter: `Q${s.quarter}`,
      unemployment: s.unemployment_rate,
      vix: s.vix,
      hpi: s.hpi_growth,
      fedFunds: s.fed_funds_rate
    }));
  }, [forecastResults]);

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950">
      {/* Header */}
      <header className="border-b border-slate-800 bg-slate-900/50 backdrop-blur-sm sticky top-0 z-50">
        <div className="container mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div className="p-2 bg-gradient-to-br from-blue-600 to-cyan-600 rounded-xl">
                <Activity className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="text-2xl font-bold text-white tracking-tight">Reg-Explain</h1>
                <p className="text-sm text-slate-400">PPNR Stress-Testing | CCAR Framework 2026</p>
              </div>
            </div>
            <div className="flex items-center gap-4">
              <Badge 
                data-testid="model-status-badge"
                variant={modelStatus?.all_ready ? "default" : "secondary"}
                className={modelStatus?.all_ready ? "bg-emerald-600" : "bg-slate-700"}
              >
                {modelStatus?.all_ready ? "Models Ready" : "Models Not Trained"}
              </Badge>
              <Select value={selectedScenario} onValueChange={setSelectedScenario}>
                <SelectTrigger data-testid="scenario-selector" className="w-48 bg-slate-800 border-slate-700">
                  <SelectValue placeholder="Select Scenario" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="baseline">📈 Baseline</SelectItem>
                  <SelectItem value="adverse">⚠️ Adverse</SelectItem>
                  <SelectItem value="severely_adverse">🔴 Severely Adverse</SelectItem>
                  <SelectItem value="custom">🎛️ Custom Black Swan</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>
        </div>
      </header>

      <main className="container mx-auto px-6 py-8">
        <Tabs defaultValue="data" className="space-y-8">
          <TabsList className="bg-slate-800/50 p-1 rounded-xl">
            <TabsTrigger data-testid="tab-data" value="data" className="data-[state=active]:bg-blue-600">
              <Database className="w-4 h-4 mr-2" />
              Data Ingestion
            </TabsTrigger>
            <TabsTrigger data-testid="tab-model" value="model" className="data-[state=active]:bg-blue-600">
              <Brain className="w-4 h-4 mr-2" />
              Model Training
            </TabsTrigger>
            <TabsTrigger data-testid="tab-forecast" value="forecast" className="data-[state=active]:bg-blue-600">
              <TrendingDown className="w-4 h-4 mr-2" />
              Capital Forecast
            </TabsTrigger>
            <TabsTrigger data-testid="tab-explain" value="explain" className="data-[state=active]:bg-blue-600">
              <Zap className="w-4 h-4 mr-2" />
              Explainability
            </TabsTrigger>
            <TabsTrigger data-testid="tab-reports" value="reports" className="data-[state=active]:bg-blue-600">
              <FileText className="w-4 h-4 mr-2" />
              Reports
            </TabsTrigger>
          </TabsList>

          {/* Data Ingestion Tab */}
          <TabsContent data-testid="data-tab-content" value="data" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Bank Data Card */}
              <Card className="bg-slate-900/50 border-slate-800">
                <CardHeader>
                  <CardTitle className="text-white flex items-center gap-2">
                    <Building2 className="w-5 h-5 text-blue-400" />
                    Bank Portfolio Data
                  </CardTitle>
                  <CardDescription>Generate synthetic or upload custom bank data</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <Button 
                    data-testid="generate-banks-btn"
                    onClick={generateSyntheticBanks}
                    disabled={loading.banks}
                    className="w-full bg-blue-600 hover:bg-blue-700"
                  >
                    {loading.banks ? <RefreshCw className="w-4 h-4 mr-2 animate-spin" /> : <Database className="w-4 h-4 mr-2" />}
                    Generate Synthetic Banks
                  </Button>
                  
                  {bankData.length > 0 && (
                    <div className="mt-4">
                      <p className="text-sm text-slate-400 mb-2">
                        Loaded {new Set(bankData.map(b => b.bank_id)).size} banks
                      </p>
                      <ScrollArea className="h-48 rounded-md border border-slate-700">
                        <div className="p-4 space-y-2">
                          {[...new Set(bankData.map(b => b.bank_id))].slice(0, 5).map(bankId => {
                            const bank = bankData.find(b => b.bank_id === bankId);
                            return (
                              <div key={bankId} className="flex justify-between text-sm">
                                <span className="text-slate-300">{bank?.bank_name || bankId}</span>
                                <span className="text-blue-400">
                                  CET1: {((bank?.cet1_ratio || 0) * 100).toFixed(1)}%
                                </span>
                              </div>
                            );
                          })}
                        </div>
                      </ScrollArea>
                    </div>
                  )}
                </CardContent>
              </Card>

              {/* Macro Data Card */}
              <Card className="bg-slate-900/50 border-slate-800">
                <CardHeader>
                  <CardTitle className="text-white flex items-center gap-2">
                    <Activity className="w-5 h-5 text-cyan-400" />
                    Macroeconomic Data (FRED)
                  </CardTitle>
                  <CardDescription>Real-time economic indicators</CardDescription>
                </CardHeader>
                <CardContent>
                  <Button 
                    data-testid="fetch-macro-btn"
                    onClick={fetchCurrentMacro}
                    className="w-full bg-cyan-600 hover:bg-cyan-700 mb-4"
                  >
                    <RefreshCw className="w-4 h-4 mr-2" />
                    Refresh FRED Data
                  </Button>
                  
                  {currentConditions && (
                    <div className="grid grid-cols-2 gap-4">
                      <div className="p-3 bg-slate-800 rounded-lg">
                        <p className="text-xs text-slate-400">Unemployment</p>
                        <p className="text-xl font-bold text-white">
                          {currentConditions.unemployment_rate?.toFixed(1)}%
                        </p>
                      </div>
                      <div className="p-3 bg-slate-800 rounded-lg">
                        <p className="text-xs text-slate-400">Fed Funds</p>
                        <p className="text-xl font-bold text-white">
                          {currentConditions.fed_funds_rate?.toFixed(2)}%
                        </p>
                      </div>
                      <div className="p-3 bg-slate-800 rounded-lg">
                        <p className="text-xs text-slate-400">VIX</p>
                        <p className="text-xl font-bold text-white">
                          {currentConditions.vix?.toFixed(1)}
                        </p>
                      </div>
                      <div className="p-3 bg-slate-800 rounded-lg">
                        <p className="text-xs text-slate-400">Yield Spread</p>
                        <p className={`text-xl font-bold ${currentConditions.yield_curve_spread < 0 ? 'text-red-400' : 'text-green-400'}`}>
                          {currentConditions.yield_curve_spread?.toFixed(2)}%
                        </p>
                      </div>
                    </div>
                  )}
                </CardContent>
              </Card>
            </div>

            {/* Scenario Info */}
            {selectedScenario && scenarios[selectedScenario] && (
              <Card className="bg-gradient-to-r from-slate-900 to-slate-800 border-blue-800/50">
                <CardContent className="p-6">
                  <div className="flex items-center justify-between">
                    <div>
                      <h3 className="text-lg font-semibold text-white">
                        {scenarios[selectedScenario].name} Scenario
                      </h3>
                      <p className="text-slate-400">{scenarios[selectedScenario].description}</p>
                    </div>
                    <div className="flex gap-6 text-center">
                      <div>
                        <p className="text-2xl font-bold text-red-400">
                          {scenarios[selectedScenario].unemployment_peak}%
                        </p>
                        <p className="text-xs text-slate-500">Unemployment Peak</p>
                      </div>
                      <div>
                        <p className="text-2xl font-bold text-amber-400">
                          {scenarios[selectedScenario].vix_peak}
                        </p>
                        <p className="text-xs text-slate-500">VIX Peak</p>
                      </div>
                      <div>
                        <p className="text-2xl font-bold text-emerald-400">
                          {scenarios[selectedScenario].hpi_decline}%
                        </p>
                        <p className="text-xs text-slate-500">HPI Decline</p>
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            )}
          </TabsContent>

          {/* Model Training Tab */}
          <TabsContent data-testid="model-tab-content" value="model" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* LSTM Config */}
              <Card className="bg-slate-900/50 border-slate-800">
                <CardHeader>
                  <CardTitle className="text-white">🔄 LSTM Configuration</CardTitle>
                  <CardDescription>Temporal model for interest rate path dependency</CardDescription>
                </CardHeader>
                <CardContent className="space-y-6">
                  <div>
                    <label className="text-sm text-slate-400">LSTM Units: {trainingConfig.lstm_units}</label>
                    <Slider
                      data-testid="lstm-units-slider"
                      value={[trainingConfig.lstm_units]}
                      onValueChange={([v]) => setTrainingConfig(prev => ({ ...prev, lstm_units: v }))}
                      min={32}
                      max={128}
                      step={8}
                      className="mt-2"
                    />
                  </div>
                  <div>
                    <label className="text-sm text-slate-400">Dropout Rate: {trainingConfig.lstm_dropout}</label>
                    <Slider
                      value={[trainingConfig.lstm_dropout * 100]}
                      onValueChange={([v]) => setTrainingConfig(prev => ({ ...prev, lstm_dropout: v / 100 }))}
                      min={10}
                      max={50}
                      step={5}
                      className="mt-2"
                    />
                  </div>
                  <div>
                    <label className="text-sm text-slate-400">Training Epochs: {trainingConfig.lstm_epochs}</label>
                    <Slider
                      value={[trainingConfig.lstm_epochs]}
                      onValueChange={([v]) => setTrainingConfig(prev => ({ ...prev, lstm_epochs: v }))}
                      min={50}
                      max={200}
                      step={10}
                      className="mt-2"
                    />
                  </div>
                </CardContent>
              </Card>

              {/* XGBoost Config */}
              <Card className="bg-slate-900/50 border-slate-800">
                <CardHeader>
                  <CardTitle className="text-white">🌲 XGBoost Configuration</CardTitle>
                  <CardDescription>Gradient boosting for static risk factors</CardDescription>
                </CardHeader>
                <CardContent className="space-y-6">
                  <div>
                    <label className="text-sm text-slate-400">N Estimators: {trainingConfig.xgb_estimators}</label>
                    <Slider
                      data-testid="xgb-estimators-slider"
                      value={[trainingConfig.xgb_estimators]}
                      onValueChange={([v]) => setTrainingConfig(prev => ({ ...prev, xgb_estimators: v }))}
                      min={100}
                      max={500}
                      step={50}
                      className="mt-2"
                    />
                  </div>
                  <div>
                    <label className="text-sm text-slate-400">Max Depth: {trainingConfig.xgb_depth}</label>
                    <Slider
                      value={[trainingConfig.xgb_depth]}
                      onValueChange={([v]) => setTrainingConfig(prev => ({ ...prev, xgb_depth: v }))}
                      min={3}
                      max={10}
                      step={1}
                      className="mt-2"
                    />
                  </div>
                  <div>
                    <label className="text-sm text-slate-400">Learning Rate: {trainingConfig.xgb_learning_rate}</label>
                    <Slider
                      value={[trainingConfig.xgb_learning_rate * 100]}
                      onValueChange={([v]) => setTrainingConfig(prev => ({ ...prev, xgb_learning_rate: v / 100 }))}
                      min={1}
                      max={20}
                      step={1}
                      className="mt-2"
                    />
                  </div>
                </CardContent>
              </Card>
            </div>

            {/* Training Button & Progress */}
            <Card className="bg-slate-900/50 border-slate-800">
              <CardContent className="p-6">
                <div className="space-y-4">
                  {trainingProgress > 0 && (
                    <div className="space-y-2">
                      <div className="flex justify-between text-sm">
                        <span className="text-slate-400">Training Progress</span>
                        <span className="text-blue-400">{trainingProgress}%</span>
                      </div>
                      <Progress value={trainingProgress} className="h-2" />
                    </div>
                  )}
                  
                  <Button 
                    data-testid="train-models-btn"
                    onClick={trainModels}
                    disabled={loading.training}
                    className="w-full bg-gradient-to-r from-blue-600 to-cyan-600 hover:from-blue-700 hover:to-cyan-700 h-12 text-lg"
                  >
                    {loading.training ? (
                      <>
                        <RefreshCw className="w-5 h-5 mr-2 animate-spin" />
                        Training Ensemble Model...
                      </>
                    ) : (
                      <>
                        <Play className="w-5 h-5 mr-2" />
                        Train LSTM-XGBoost Ensemble
                      </>
                    )}
                  </Button>
                </div>

                {modelStatus?.all_ready && modelStatus?.feature_importance && (
                  <div className="mt-6">
                    <h4 className="text-sm font-medium text-slate-300 mb-3">Feature Importance</h4>
                    <div className="space-y-2">
                      {Object.entries(modelStatus.feature_importance)
                        .sort(([,a], [,b]) => b - a)
                        .map(([feature, importance]) => (
                          <div key={feature} className="flex items-center gap-2">
                            <span className="text-xs text-slate-400 w-32 truncate">
                              {feature.replace(/_/g, ' ')}
                            </span>
                            <div className="flex-1 h-2 bg-slate-700 rounded-full overflow-hidden">
                              <div 
                                className="h-full bg-gradient-to-r from-blue-500 to-cyan-500 rounded-full"
                                style={{ width: `${importance * 100}%` }}
                              />
                            </div>
                            <span className="text-xs text-slate-500 w-12 text-right">
                              {(importance * 100).toFixed(1)}%
                            </span>
                          </div>
                        ))}
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>
          </TabsContent>

          {/* Capital Forecast Tab */}
          <TabsContent data-testid="forecast-tab-content" value="forecast" className="space-y-6">
            <Button 
              data-testid="run-forecast-btn"
              onClick={runForecast}
              disabled={loading.forecast}
              className="w-full bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-700 hover:to-teal-700 h-14 text-lg"
            >
              {loading.forecast ? (
                <>
                  <RefreshCw className="w-5 h-5 mr-2 animate-spin" />
                  Running 9-Quarter Forecast...
                </>
              ) : (
                <>
                  <TrendingDown className="w-5 h-5 mr-2" />
                  Run Capital Erosion Forecast
                </>
              )}
            </Button>

            {metrics && (
              <>
                {/* Key Metrics */}
                <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
                  <Card className={`bg-slate-900/50 border-l-4 ${metrics.minCet1 > 7 ? 'border-emerald-500' : metrics.minCet1 > 4.5 ? 'border-amber-500' : 'border-red-500'}`}>
                    <CardContent className="p-4">
                      <p className="text-xs text-slate-400 uppercase tracking-wide">Minimum CET1</p>
                      <p className={`text-3xl font-bold font-mono ${metrics.minCet1 > 7 ? 'text-emerald-400' : metrics.minCet1 > 4.5 ? 'text-amber-400' : 'text-red-400'}`}>
                        {metrics.minCet1.toFixed(1)}%
                      </p>
                    </CardContent>
                  </Card>
                  
                  <Card className="bg-slate-900/50 border-l-4 border-amber-500">
                    <CardContent className="p-4">
                      <p className="text-xs text-slate-400 uppercase tracking-wide">Capital Decline</p>
                      <p className="text-3xl font-bold font-mono text-amber-400">
                        {metrics.decline.toFixed(1)}%
                      </p>
                    </CardContent>
                  </Card>
                  
                  <Card className={`bg-slate-900/50 border-l-4 ${metrics.breachRate === 0 ? 'border-emerald-500' : metrics.breachRate < 30 ? 'border-amber-500' : 'border-red-500'}`}>
                    <CardContent className="p-4">
                      <p className="text-xs text-slate-400 uppercase tracking-wide">Banks Breaching</p>
                      <p className={`text-3xl font-bold font-mono ${metrics.breachRate === 0 ? 'text-emerald-400' : metrics.breachRate < 30 ? 'text-amber-400' : 'text-red-400'}`}>
                        {metrics.breachCount}/{metrics.totalBanks}
                      </p>
                    </CardContent>
                  </Card>
                  
                  <Card className="bg-slate-900/50 border-l-4 border-red-500">
                    <CardContent className="p-4">
                      <p className="text-xs text-slate-400 uppercase tracking-wide">Total Losses</p>
                      <p className="text-3xl font-bold font-mono text-red-400">
                        ${metrics.totalLosses.toFixed(1)}B
                      </p>
                    </CardContent>
                  </Card>
                </div>

                {/* Breach Alert */}
                {metrics.breachCount > 0 && (
                  <Alert data-testid="breach-alert" className="bg-red-950/50 border-red-800">
                    <AlertTriangle className="h-5 w-5 text-red-400" />
                    <AlertTitle className="text-red-300">Capital Breach Alert</AlertTitle>
                    <AlertDescription className="text-red-200">
                      {metrics.breachCount} of {metrics.totalBanks} banks projected to breach CET1 minimum. 
                      Immediate supervisory review recommended per SR 11-7.
                    </AlertDescription>
                  </Alert>
                )}

                {/* Charts */}
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  {/* CET1 Trajectory */}
                  <Card className="bg-slate-900/50 border-slate-800">
                    <CardHeader>
                      <CardTitle className="text-white">CET1 Capital Erosion Map</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <ResponsiveContainer width="100%" height={300}>
                        <AreaChart data={getTrajectoryData()}>
                          <defs>
                            <linearGradient id="cet1Gradient" x1="0" y1="0" x2="0" y2="1">
                              <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3}/>
                              <stop offset="95%" stopColor="#3b82f6" stopOpacity={0}/>
                            </linearGradient>
                          </defs>
                          <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                          <XAxis dataKey="quarter" stroke="#94a3b8" />
                          <YAxis stroke="#94a3b8" domain={[0, 'auto']} />
                          <Tooltip 
                            contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155' }}
                            labelStyle={{ color: '#f8fafc' }}
                          />
                          <Area 
                            type="monotone" 
                            dataKey="avgCet1" 
                            stroke="#3b82f6" 
                            fill="url(#cet1Gradient)"
                            strokeWidth={2}
                          />
                          <Line 
                            type="monotone" 
                            dataKey="minCet1" 
                            stroke="#ef4444" 
                            strokeDasharray="5 5"
                            dot={false}
                          />
                          {/* Regulatory minimum line */}
                          <Line
                            type="monotone"
                            dataKey={() => 7}
                            stroke="#f59e0b"
                            strokeDasharray="3 3"
                            dot={false}
                          />
                        </AreaChart>
                      </ResponsiveContainer>
                    </CardContent>
                  </Card>

                  {/* Macro Scenario */}
                  <Card className="bg-slate-900/50 border-slate-800">
                    <CardHeader>
                      <CardTitle className="text-white">Macro Stress Path</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <ResponsiveContainer width="100%" height={300}>
                        <ComposedChart data={getScenarioData()}>
                          <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                          <XAxis dataKey="quarter" stroke="#94a3b8" />
                          <YAxis yAxisId="left" stroke="#94a3b8" />
                          <YAxis yAxisId="right" orientation="right" stroke="#94a3b8" />
                          <Tooltip 
                            contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155' }}
                          />
                          <Legend />
                          <Bar yAxisId="left" dataKey="unemployment" fill="#ef4444" name="Unemployment %" />
                          <Line yAxisId="right" type="monotone" dataKey="vix" stroke="#f59e0b" name="VIX" strokeWidth={2} />
                          <Line yAxisId="left" type="monotone" dataKey="fedFunds" stroke="#22c55e" name="Fed Funds %" strokeWidth={2} />
                        </ComposedChart>
                      </ResponsiveContainer>
                    </CardContent>
                  </Card>
                </div>
              </>
            )}
          </TabsContent>

          {/* Explainability Tab */}
          <TabsContent data-testid="explain-tab-content" value="explain" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* SHAP Analysis */}
              <Card className="bg-slate-900/50 border-slate-800">
                <CardHeader>
                  <CardTitle className="text-white flex items-center gap-2">
                    <Zap className="w-5 h-5 text-purple-400" />
                    Global Feature Importance (SHAP)
                  </CardTitle>
                  <CardDescription>SR 11-7 Model Risk Management Compliance</CardDescription>
                </CardHeader>
                <CardContent>
                  <Button 
                    data-testid="compute-shap-btn"
                    onClick={fetchShapExplanation}
                    disabled={loading.shap || !modelStatus?.all_ready}
                    className="w-full bg-purple-600 hover:bg-purple-700 mb-4"
                  >
                    {loading.shap ? <RefreshCw className="w-4 h-4 mr-2 animate-spin" /> : <Zap className="w-4 h-4 mr-2" />}
                    Compute SHAP Values
                  </Button>
                  
                  {shapExplanation && (
                    <div className="space-y-3">
                      {Object.entries(shapExplanation.global_importance)
                        .sort(([,a], [,b]) => b - a)
                        .map(([feature, importance], idx) => {
                          const pct = (importance / Object.values(shapExplanation.global_importance).reduce((a,b) => a+b, 0)) * 100;
                          return (
                            <div key={feature}>
                              <div className="flex justify-between text-sm mb-1">
                                <span className="text-slate-300">{feature.replace(/_/g, ' ')}</span>
                                <span className="text-purple-400">{pct.toFixed(1)}%</span>
                              </div>
                              <div className="h-2 bg-slate-700 rounded-full overflow-hidden">
                                <div 
                                  className="h-full bg-gradient-to-r from-purple-500 to-pink-500 rounded-full transition-all duration-500"
                                  style={{ width: `${pct}%` }}
                                />
                              </div>
                            </div>
                          );
                        })}
                    </div>
                  )}
                </CardContent>
              </Card>

              {/* AI Explanation */}
              <Card className="bg-slate-900/50 border-slate-800">
                <CardHeader>
                  <CardTitle className="text-white flex items-center gap-2">
                    <Brain className="w-5 h-5 text-cyan-400" />
                    AI-Powered Analysis (Gemini 3 Flash)
                  </CardTitle>
                  <CardDescription>Natural language interpretation of model results</CardDescription>
                </CardHeader>
                <CardContent>
                  <Button 
                    data-testid="generate-ai-explanation-btn"
                    onClick={fetchAiExplanation}
                    disabled={loading.ai || !modelStatus?.all_ready}
                    className="w-full bg-cyan-600 hover:bg-cyan-700 mb-4"
                  >
                    {loading.ai ? <RefreshCw className="w-4 h-4 mr-2 animate-spin" /> : <Brain className="w-4 h-4 mr-2" />}
                    Generate AI Explanation
                  </Button>
                  
                  {aiExplanation && (
                    <div className="p-4 bg-slate-800 rounded-lg border-l-4 border-cyan-500">
                      <p className="text-slate-200 leading-relaxed whitespace-pre-wrap">
                        {aiExplanation}
                      </p>
                    </div>
                  )}
                </CardContent>
              </Card>
            </div>

            {/* Compliance Note */}
            <Alert className="bg-slate-800/50 border-slate-700">
              <Shield className="h-5 w-5 text-blue-400" />
              <AlertTitle className="text-slate-200">SR 11-7 Model Risk Management</AlertTitle>
              <AlertDescription className="text-slate-400">
                This analysis complies with Federal Reserve SR 11-7 guidelines for model risk management, 
                providing transparent feature attribution and scenario-specific explanations for regulatory review.
              </AlertDescription>
            </Alert>
          </TabsContent>

          {/* Reports Tab */}
          <TabsContent data-testid="reports-tab-content" value="reports" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              <Card className="bg-slate-900/50 border-slate-800">
                <CardHeader>
                  <CardTitle className="text-white">CET1 Trajectory Report</CardTitle>
                </CardHeader>
                <CardContent>
                  <Button 
                    data-testid="download-trajectory-btn"
                    disabled={!forecastResults}
                    className="w-full bg-blue-600 hover:bg-blue-700"
                    onClick={() => {
                      if (forecastResults?.trajectory) {
                        const csv = Object.keys(forecastResults.trajectory[0]).join(',') + '\n' +
                          forecastResults.trajectory.map(row => Object.values(row).join(',')).join('\n');
                        const blob = new Blob([csv], { type: 'text/csv' });
                        const url = URL.createObjectURL(blob);
                        const a = document.createElement('a');
                        a.href = url;
                        a.download = `cet1_trajectory_${new Date().toISOString().split('T')[0]}.csv`;
                        a.click();
                      }
                    }}
                  >
                    <Download className="w-4 h-4 mr-2" />
                    Download CSV
                  </Button>
                </CardContent>
              </Card>

              <Card className="bg-slate-900/50 border-slate-800">
                <CardHeader>
                  <CardTitle className="text-white">Executive Summary</CardTitle>
                </CardHeader>
                <CardContent>
                  <Button 
                    data-testid="download-summary-btn"
                    disabled={!forecastResults}
                    className="w-full bg-emerald-600 hover:bg-emerald-700"
                  >
                    <Download className="w-4 h-4 mr-2" />
                    Download PDF
                  </Button>
                </CardContent>
              </Card>

              <Card className="bg-slate-900/50 border-slate-800">
                <CardHeader>
                  <CardTitle className="text-white">SHAP Analysis</CardTitle>
                </CardHeader>
                <CardContent>
                  <Button 
                    data-testid="download-shap-btn"
                    disabled={!shapExplanation}
                    className="w-full bg-purple-600 hover:bg-purple-700"
                  >
                    <Download className="w-4 h-4 mr-2" />
                    Download Report
                  </Button>
                </CardContent>
              </Card>
            </div>

            {forecastResults?.trajectory && (
              <Card className="bg-slate-900/50 border-slate-800">
                <CardHeader>
                  <CardTitle className="text-white">Bank-Level Results Summary</CardTitle>
                </CardHeader>
                <CardContent>
                  <ScrollArea className="h-96">
                    <table className="w-full text-sm">
                      <thead className="sticky top-0 bg-slate-800">
                        <tr>
                          <th className="text-left p-3 text-slate-400">Bank</th>
                          <th className="text-right p-3 text-slate-400">Initial CET1</th>
                          <th className="text-right p-3 text-slate-400">Min CET1</th>
                          <th className="text-right p-3 text-slate-400">Final CET1</th>
                          <th className="text-right p-3 text-slate-400">Total Losses</th>
                          <th className="text-center p-3 text-slate-400">Breach</th>
                        </tr>
                      </thead>
                      <tbody>
                        {[...new Set(forecastResults.trajectory.map(t => t.bank_id))].map(bankId => {
                          const bankData = forecastResults.trajectory.filter(t => t.bank_id === bankId);
                          const initial = bankData.find(t => t.quarter === 1);
                          const final = bankData.find(t => t.quarter === 9);
                          const minCet1 = Math.min(...bankData.map(t => t.cet1_ratio));
                          const totalLosses = bankData.reduce((sum, t) => sum + t.credit_losses, 0);
                          const hasBreach = bankData.some(t => t.breach);
                          
                          return (
                            <tr key={bankId} className="border-t border-slate-700 hover:bg-slate-800/50">
                              <td className="p-3 text-slate-300">{bankData[0]?.bank_name || bankId}</td>
                              <td className="p-3 text-right text-slate-300">
                                {((initial?.cet1_ratio || 0) * 100).toFixed(2)}%
                              </td>
                              <td className={`p-3 text-right ${minCet1 < 0.045 ? 'text-red-400' : minCet1 < 0.07 ? 'text-amber-400' : 'text-emerald-400'}`}>
                                {(minCet1 * 100).toFixed(2)}%
                              </td>
                              <td className="p-3 text-right text-slate-300">
                                {((final?.cet1_ratio || 0) * 100).toFixed(2)}%
                              </td>
                              <td className="p-3 text-right text-red-400">
                                ${(totalLosses / 1e9).toFixed(2)}B
                              </td>
                              <td className="p-3 text-center">
                                {hasBreach ? (
                                  <Badge className="bg-red-600">Breach</Badge>
                                ) : (
                                  <Badge className="bg-emerald-600">Safe</Badge>
                                )}
                              </td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </ScrollArea>
                </CardContent>
              </Card>
            )}
          </TabsContent>
        </Tabs>
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-800 py-6 mt-12">
        <div className="container mx-auto px-6 text-center text-slate-500 text-sm">
          <p>Reg-Explain | CCAR Framework 2026 | PPNR Stress-Testing via Hybrid LSTM-XGBoost with SHAP Interpretability</p>
          <p className="mt-1">Compliant with SR 11-7 Model Risk Management Guidelines</p>
        </div>
      </footer>
    </div>
  );
}

export default App;
