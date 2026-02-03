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
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, 
  ResponsiveContainer, BarChart, Bar, AreaChart, Area, ComposedChart
} from 'recharts';
import { 
  TrendingDown, AlertTriangle, Activity, Database, 
  Brain, FileText, Download, RefreshCw, Play,
  Building2, Shield, Zap
} from 'lucide-react';
import axios from 'axios';
import './App.css';

const API_URL = process.env.REACT_APP_BACKEND_URL;

// Bloomberg Terminal Color Palette
const COLORS = {
  bgPrimary: '#0D1117',
  bgSurface: '#161B22',
  bgElevated: '#1C2128',
  accentCyan: '#00D1FF',
  accentGreen: '#00FF41',
  accentRed: '#FF3131',
  accentYellow: '#FFB800',
  textPrimary: '#E6EDF3',
  textSecondary: '#8B949E',
  border: '#30363D'
};

function App() {
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
  
  const [trainingConfig, setTrainingConfig] = useState({
    lstm_units: 64,
    lstm_dropout: 0.2,
    lstm_epochs: 100,
    xgb_estimators: 200,
    xgb_depth: 6,
    xgb_learning_rate: 0.05
  });

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
    <div className="min-h-screen" style={{ backgroundColor: COLORS.bgPrimary }}>
      {/* Header */}
      <header className="border-b sticky top-0 z-50" style={{ borderColor: COLORS.border, backgroundColor: COLORS.bgSurface }}>
        <div className="container mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div className="p-2 rounded-lg" style={{ backgroundColor: COLORS.accentCyan }}>
                <Activity className="w-6 h-6" style={{ color: COLORS.bgPrimary }} />
              </div>
              <div>
                <h1 className="text-2xl font-bold tracking-tight" style={{ color: COLORS.textPrimary }}>
                  REG-EXPLAIN
                </h1>
                <p className="text-sm" style={{ color: COLORS.textSecondary }}>
                  PPNR Stress-Testing | CCAR Framework 2026
                </p>
              </div>
            </div>
            <div className="flex items-center gap-4">
              <Badge 
                data-testid="model-status-badge"
                className="px-3 py-1 font-mono text-xs"
                style={{ 
                  backgroundColor: modelStatus?.all_ready ? 'rgba(0, 255, 65, 0.15)' : 'rgba(139, 148, 158, 0.15)',
                  color: modelStatus?.all_ready ? COLORS.accentGreen : COLORS.textSecondary,
                  border: `1px solid ${modelStatus?.all_ready ? 'rgba(0, 255, 65, 0.3)' : COLORS.border}`
                }}
              >
                {modelStatus?.all_ready ? "● MODELS READY" : "○ NOT TRAINED"}
              </Badge>
              <Select value={selectedScenario} onValueChange={setSelectedScenario}>
                <SelectTrigger 
                  data-testid="scenario-selector" 
                  className="w-52 font-mono text-sm"
                  style={{ backgroundColor: COLORS.bgElevated, borderColor: COLORS.border, color: COLORS.textPrimary }}
                >
                  <SelectValue placeholder="Select Scenario" />
                </SelectTrigger>
                <SelectContent style={{ backgroundColor: COLORS.bgSurface, borderColor: COLORS.border }}>
                  <SelectItem value="baseline">BASELINE</SelectItem>
                  <SelectItem value="adverse">ADVERSE</SelectItem>
                  <SelectItem value="severely_adverse">SEVERELY ADVERSE</SelectItem>
                  <SelectItem value="custom">CUSTOM BLACK SWAN</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>
        </div>
      </header>

      <main className="container mx-auto px-6 py-8">
        <Tabs defaultValue="data" className="space-y-8">
          <TabsList className="p-1 rounded-lg" style={{ backgroundColor: COLORS.bgSurface, borderColor: COLORS.border }}>
            <TabsTrigger data-testid="tab-data" value="data" className="font-mono text-sm px-4">
              <Database className="w-4 h-4 mr-2" />
              DATA
            </TabsTrigger>
            <TabsTrigger data-testid="tab-model" value="model" className="font-mono text-sm px-4">
              <Brain className="w-4 h-4 mr-2" />
              MODEL
            </TabsTrigger>
            <TabsTrigger data-testid="tab-forecast" value="forecast" className="font-mono text-sm px-4">
              <TrendingDown className="w-4 h-4 mr-2" />
              FORECAST
            </TabsTrigger>
            <TabsTrigger data-testid="tab-explain" value="explain" className="font-mono text-sm px-4">
              <Zap className="w-4 h-4 mr-2" />
              EXPLAIN
            </TabsTrigger>
            <TabsTrigger data-testid="tab-reports" value="reports" className="font-mono text-sm px-4">
              <FileText className="w-4 h-4 mr-2" />
              REPORTS
            </TabsTrigger>
          </TabsList>

          {/* Data Ingestion Tab */}
          <TabsContent data-testid="data-tab-content" value="data" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <Card style={{ backgroundColor: COLORS.bgSurface, borderColor: COLORS.border }}>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2" style={{ color: COLORS.textPrimary }}>
                    <Building2 className="w-5 h-5" style={{ color: COLORS.accentCyan }} />
                    BANK PORTFOLIO DATA
                  </CardTitle>
                  <CardDescription style={{ color: COLORS.textSecondary }}>
                    Generate synthetic or upload custom bank data
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <Button 
                    data-testid="generate-banks-btn"
                    onClick={generateSyntheticBanks}
                    disabled={loading.banks}
                    className="w-full font-mono"
                    style={{ backgroundColor: COLORS.accentCyan, color: COLORS.bgPrimary }}
                  >
                    {loading.banks ? <RefreshCw className="w-4 h-4 mr-2 animate-spin" /> : <Database className="w-4 h-4 mr-2" />}
                    GENERATE SYNTHETIC BANKS
                  </Button>
                  
                  {bankData.length > 0 && (
                    <div className="mt-4">
                      <p className="text-sm font-mono mb-2" style={{ color: COLORS.textSecondary }}>
                        LOADED: {new Set(bankData.map(b => b.bank_id)).size} BANKS
                      </p>
                      <ScrollArea className="h-48 rounded border" style={{ borderColor: COLORS.border }}>
                        <div className="p-4 space-y-2">
                          {[...new Set(bankData.map(b => b.bank_id))].slice(0, 5).map(bankId => {
                            const bank = bankData.find(b => b.bank_id === bankId);
                            return (
                              <div key={bankId} className="flex justify-between text-sm font-mono">
                                <span style={{ color: COLORS.textPrimary }}>{bank?.bank_name || bankId}</span>
                                <span style={{ color: COLORS.accentCyan }}>
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

              <Card style={{ backgroundColor: COLORS.bgSurface, borderColor: COLORS.border }}>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2" style={{ color: COLORS.textPrimary }}>
                    <Activity className="w-5 h-5" style={{ color: COLORS.accentCyan }} />
                    MACROECONOMIC DATA (FRED)
                  </CardTitle>
                  <CardDescription style={{ color: COLORS.textSecondary }}>
                    Real-time economic indicators
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <Button 
                    data-testid="fetch-macro-btn"
                    onClick={fetchCurrentMacro}
                    className="w-full font-mono mb-4"
                    style={{ backgroundColor: COLORS.accentCyan, color: COLORS.bgPrimary }}
                  >
                    <RefreshCw className="w-4 h-4 mr-2" />
                    REFRESH FRED DATA
                  </Button>
                  
                  {currentConditions && (
                    <div className="grid grid-cols-2 gap-3">
                      <div className="p-3 rounded" style={{ backgroundColor: COLORS.bgElevated, borderLeft: `3px solid ${COLORS.accentCyan}` }}>
                        <p className="text-xs font-mono" style={{ color: COLORS.textSecondary }}>UNEMPLOYMENT</p>
                        <p className="text-xl font-bold font-mono" style={{ color: COLORS.textPrimary }}>
                          {currentConditions.unemployment_rate?.toFixed(1)}%
                        </p>
                      </div>
                      <div className="p-3 rounded" style={{ backgroundColor: COLORS.bgElevated, borderLeft: `3px solid ${COLORS.accentCyan}` }}>
                        <p className="text-xs font-mono" style={{ color: COLORS.textSecondary }}>FED FUNDS</p>
                        <p className="text-xl font-bold font-mono" style={{ color: COLORS.textPrimary }}>
                          {currentConditions.fed_funds_rate?.toFixed(2)}%
                        </p>
                      </div>
                      <div className="p-3 rounded" style={{ backgroundColor: COLORS.bgElevated, borderLeft: `3px solid ${COLORS.accentYellow}` }}>
                        <p className="text-xs font-mono" style={{ color: COLORS.textSecondary }}>VIX</p>
                        <p className="text-xl font-bold font-mono" style={{ color: COLORS.textPrimary }}>
                          {currentConditions.vix?.toFixed(1)}
                        </p>
                      </div>
                      <div className="p-3 rounded" style={{ backgroundColor: COLORS.bgElevated, borderLeft: `3px solid ${currentConditions.yield_curve_spread < 0 ? COLORS.accentRed : COLORS.accentGreen}` }}>
                        <p className="text-xs font-mono" style={{ color: COLORS.textSecondary }}>YIELD SPREAD</p>
                        <p className="text-xl font-bold font-mono" style={{ color: currentConditions.yield_curve_spread < 0 ? COLORS.accentRed : COLORS.accentGreen }}>
                          {currentConditions.yield_curve_spread?.toFixed(2)}%
                        </p>
                      </div>
                    </div>
                  )}
                </CardContent>
              </Card>
            </div>

            {selectedScenario && scenarios[selectedScenario] && (
              <Card style={{ backgroundColor: COLORS.bgSurface, borderColor: COLORS.accentCyan, borderWidth: '1px' }}>
                <CardContent className="p-6">
                  <div className="flex items-center justify-between">
                    <div>
                      <h3 className="text-lg font-bold font-mono" style={{ color: COLORS.accentCyan }}>
                        {scenarios[selectedScenario].name.toUpperCase()} SCENARIO
                      </h3>
                      <p style={{ color: COLORS.textSecondary }}>{scenarios[selectedScenario].description}</p>
                    </div>
                    <div className="flex gap-8 text-center">
                      <div>
                        <p className="text-2xl font-bold font-mono" style={{ color: COLORS.accentRed }}>
                          {scenarios[selectedScenario].unemployment_peak}%
                        </p>
                        <p className="text-xs font-mono" style={{ color: COLORS.textSecondary }}>UNEMP PEAK</p>
                      </div>
                      <div>
                        <p className="text-2xl font-bold font-mono" style={{ color: COLORS.accentYellow }}>
                          {scenarios[selectedScenario].vix_peak}
                        </p>
                        <p className="text-xs font-mono" style={{ color: COLORS.textSecondary }}>VIX PEAK</p>
                      </div>
                      <div>
                        <p className="text-2xl font-bold font-mono" style={{ color: COLORS.accentGreen }}>
                          {scenarios[selectedScenario].hpi_decline}%
                        </p>
                        <p className="text-xs font-mono" style={{ color: COLORS.textSecondary }}>HPI DECLINE</p>
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
              <Card style={{ backgroundColor: COLORS.bgSurface, borderColor: COLORS.border }}>
                <CardHeader>
                  <CardTitle style={{ color: COLORS.textPrimary }}>LSTM CONFIGURATION</CardTitle>
                  <CardDescription style={{ color: COLORS.textSecondary }}>Temporal model for interest rate paths</CardDescription>
                </CardHeader>
                <CardContent className="space-y-6">
                  <div>
                    <label className="text-sm font-mono" style={{ color: COLORS.textSecondary }}>
                      LSTM UNITS: <span style={{ color: COLORS.accentCyan }}>{trainingConfig.lstm_units}</span>
                    </label>
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
                    <label className="text-sm font-mono" style={{ color: COLORS.textSecondary }}>
                      DROPOUT: <span style={{ color: COLORS.accentCyan }}>{trainingConfig.lstm_dropout}</span>
                    </label>
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
                    <label className="text-sm font-mono" style={{ color: COLORS.textSecondary }}>
                      EPOCHS: <span style={{ color: COLORS.accentCyan }}>{trainingConfig.lstm_epochs}</span>
                    </label>
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

              <Card style={{ backgroundColor: COLORS.bgSurface, borderColor: COLORS.border }}>
                <CardHeader>
                  <CardTitle style={{ color: COLORS.textPrimary }}>XGBOOST CONFIGURATION</CardTitle>
                  <CardDescription style={{ color: COLORS.textSecondary }}>Gradient boosting for static risk factors</CardDescription>
                </CardHeader>
                <CardContent className="space-y-6">
                  <div>
                    <label className="text-sm font-mono" style={{ color: COLORS.textSecondary }}>
                      ESTIMATORS: <span style={{ color: COLORS.accentCyan }}>{trainingConfig.xgb_estimators}</span>
                    </label>
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
                    <label className="text-sm font-mono" style={{ color: COLORS.textSecondary }}>
                      MAX DEPTH: <span style={{ color: COLORS.accentCyan }}>{trainingConfig.xgb_depth}</span>
                    </label>
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
                    <label className="text-sm font-mono" style={{ color: COLORS.textSecondary }}>
                      LEARNING RATE: <span style={{ color: COLORS.accentCyan }}>{trainingConfig.xgb_learning_rate}</span>
                    </label>
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

            <Card style={{ backgroundColor: COLORS.bgSurface, borderColor: COLORS.border }}>
              <CardContent className="p-6">
                <div className="space-y-4">
                  {trainingProgress > 0 && (
                    <div className="space-y-2">
                      <div className="flex justify-between text-sm font-mono">
                        <span style={{ color: COLORS.textSecondary }}>TRAINING PROGRESS</span>
                        <span style={{ color: COLORS.accentCyan }}>{trainingProgress}%</span>
                      </div>
                      <Progress value={trainingProgress} className="h-2" style={{ backgroundColor: COLORS.bgElevated }} />
                    </div>
                  )}
                  
                  <Button 
                    data-testid="train-models-btn"
                    onClick={trainModels}
                    disabled={loading.training}
                    className="w-full h-14 text-lg font-mono font-bold"
                    style={{ backgroundColor: COLORS.accentCyan, color: COLORS.bgPrimary }}
                  >
                    {loading.training ? (
                      <>
                        <RefreshCw className="w-5 h-5 mr-2 animate-spin" />
                        TRAINING ENSEMBLE MODEL...
                      </>
                    ) : (
                      <>
                        <Play className="w-5 h-5 mr-2" />
                        TRAIN LSTM-XGBOOST ENSEMBLE
                      </>
                    )}
                  </Button>
                </div>

                {modelStatus?.all_ready && modelStatus?.feature_importance && (
                  <div className="mt-6">
                    <h4 className="text-sm font-mono mb-3" style={{ color: COLORS.textSecondary }}>FEATURE IMPORTANCE</h4>
                    <div className="space-y-2">
                      {Object.entries(modelStatus.feature_importance)
                        .sort(([,a], [,b]) => b - a)
                        .map(([feature, importance]) => (
                          <div key={feature} className="flex items-center gap-2">
                            <span className="text-xs font-mono w-36 truncate" style={{ color: COLORS.textSecondary }}>
                              {feature.toUpperCase().replace(/_/g, ' ')}
                            </span>
                            <div className="flex-1 h-2 rounded" style={{ backgroundColor: COLORS.bgElevated }}>
                              <div 
                                className="h-full rounded"
                                style={{ width: `${importance * 100}%`, backgroundColor: COLORS.accentCyan }}
                              />
                            </div>
                            <span className="text-xs font-mono w-12 text-right" style={{ color: COLORS.accentCyan }}>
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
              className="w-full h-16 text-lg font-mono font-bold"
              style={{ backgroundColor: COLORS.accentGreen, color: COLORS.bgPrimary }}
            >
              {loading.forecast ? (
                <>
                  <RefreshCw className="w-5 h-5 mr-2 animate-spin" />
                  RUNNING 9-QUARTER FORECAST...
                </>
              ) : (
                <>
                  <TrendingDown className="w-5 h-5 mr-2" />
                  RUN CAPITAL EROSION FORECAST
                </>
              )}
            </Button>

            {metrics && (
              <>
                <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
                  <Card style={{ backgroundColor: COLORS.bgSurface, borderLeft: `4px solid ${metrics.minCet1 > 7 ? COLORS.accentGreen : metrics.minCet1 > 4.5 ? COLORS.accentYellow : COLORS.accentRed}` }}>
                    <CardContent className="p-4">
                      <p className="text-xs font-mono" style={{ color: COLORS.textSecondary }}>MINIMUM CET1</p>
                      <p className="text-3xl font-bold font-mono" style={{ color: metrics.minCet1 > 7 ? COLORS.accentGreen : metrics.minCet1 > 4.5 ? COLORS.accentYellow : COLORS.accentRed }}>
                        {metrics.minCet1.toFixed(1)}%
                      </p>
                    </CardContent>
                  </Card>
                  
                  <Card style={{ backgroundColor: COLORS.bgSurface, borderLeft: `4px solid ${COLORS.accentYellow}` }}>
                    <CardContent className="p-4">
                      <p className="text-xs font-mono" style={{ color: COLORS.textSecondary }}>CAPITAL DECLINE</p>
                      <p className="text-3xl font-bold font-mono" style={{ color: COLORS.accentYellow }}>
                        {metrics.decline.toFixed(1)}%
                      </p>
                    </CardContent>
                  </Card>
                  
                  <Card style={{ backgroundColor: COLORS.bgSurface, borderLeft: `4px solid ${metrics.breachRate === 0 ? COLORS.accentGreen : metrics.breachRate < 30 ? COLORS.accentYellow : COLORS.accentRed}` }}>
                    <CardContent className="p-4">
                      <p className="text-xs font-mono" style={{ color: COLORS.textSecondary }}>BANKS BREACHING</p>
                      <p className="text-3xl font-bold font-mono" style={{ color: metrics.breachRate === 0 ? COLORS.accentGreen : metrics.breachRate < 30 ? COLORS.accentYellow : COLORS.accentRed }}>
                        {metrics.breachCount}/{metrics.totalBanks}
                      </p>
                    </CardContent>
                  </Card>
                  
                  <Card style={{ backgroundColor: COLORS.bgSurface, borderLeft: `4px solid ${COLORS.accentRed}` }}>
                    <CardContent className="p-4">
                      <p className="text-xs font-mono" style={{ color: COLORS.textSecondary }}>TOTAL LOSSES</p>
                      <p className="text-3xl font-bold font-mono" style={{ color: COLORS.accentRed }}>
                        ${metrics.totalLosses.toFixed(1)}B
                      </p>
                    </CardContent>
                  </Card>
                </div>

                {metrics.breachCount > 0 && (
                  <Alert data-testid="breach-alert" className="pulse-alert" style={{ backgroundColor: 'rgba(255, 49, 49, 0.1)', borderColor: COLORS.accentRed }}>
                    <AlertTriangle className="h-5 w-5" style={{ color: COLORS.accentRed }} />
                    <AlertTitle className="font-mono" style={{ color: COLORS.accentRed }}>⚠ CAPITAL BREACH ALERT</AlertTitle>
                    <AlertDescription style={{ color: COLORS.textPrimary }}>
                      {metrics.breachCount} of {metrics.totalBanks} banks projected to breach CET1 minimum. 
                      Immediate supervisory review recommended per SR 11-7.
                    </AlertDescription>
                  </Alert>
                )}

                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  <Card style={{ backgroundColor: COLORS.bgSurface, borderColor: COLORS.border }}>
                    <CardHeader>
                      <CardTitle className="font-mono" style={{ color: COLORS.textPrimary }}>CET1 CAPITAL EROSION MAP</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <ResponsiveContainer width="100%" height={300}>
                        <AreaChart data={getTrajectoryData()}>
                          <defs>
                            <linearGradient id="cet1Gradient" x1="0" y1="0" x2="0" y2="1">
                              <stop offset="5%" stopColor={COLORS.accentCyan} stopOpacity={0.3}/>
                              <stop offset="95%" stopColor={COLORS.accentCyan} stopOpacity={0}/>
                            </linearGradient>
                          </defs>
                          <CartesianGrid strokeDasharray="3 3" stroke={COLORS.border} />
                          <XAxis dataKey="quarter" stroke={COLORS.textSecondary} />
                          <YAxis stroke={COLORS.textSecondary} domain={[0, 'auto']} />
                          <Tooltip 
                            contentStyle={{ backgroundColor: COLORS.bgSurface, border: `1px solid ${COLORS.border}` }}
                            labelStyle={{ color: COLORS.textPrimary }}
                          />
                          <Area 
                            type="monotone" 
                            dataKey="avgCet1" 
                            stroke={COLORS.accentCyan} 
                            fill="url(#cet1Gradient)"
                            strokeWidth={2}
                          />
                          <Line 
                            type="monotone" 
                            dataKey="minCet1" 
                            stroke={COLORS.accentRed} 
                            strokeDasharray="5 5"
                            dot={false}
                          />
                          <Line
                            type="monotone"
                            dataKey={() => 7}
                            stroke={COLORS.accentYellow}
                            strokeDasharray="3 3"
                            dot={false}
                          />
                        </AreaChart>
                      </ResponsiveContainer>
                    </CardContent>
                  </Card>

                  <Card style={{ backgroundColor: COLORS.bgSurface, borderColor: COLORS.border }}>
                    <CardHeader>
                      <CardTitle className="font-mono" style={{ color: COLORS.textPrimary }}>MACRO STRESS PATH</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <ResponsiveContainer width="100%" height={300}>
                        <ComposedChart data={getScenarioData()}>
                          <CartesianGrid strokeDasharray="3 3" stroke={COLORS.border} />
                          <XAxis dataKey="quarter" stroke={COLORS.textSecondary} />
                          <YAxis yAxisId="left" stroke={COLORS.textSecondary} />
                          <YAxis yAxisId="right" orientation="right" stroke={COLORS.textSecondary} />
                          <Tooltip 
                            contentStyle={{ backgroundColor: COLORS.bgSurface, border: `1px solid ${COLORS.border}` }}
                          />
                          <Legend />
                          <Bar yAxisId="left" dataKey="unemployment" fill={COLORS.accentRed} name="Unemployment %" />
                          <Line yAxisId="right" type="monotone" dataKey="vix" stroke={COLORS.accentYellow} name="VIX" strokeWidth={2} />
                          <Line yAxisId="left" type="monotone" dataKey="fedFunds" stroke={COLORS.accentGreen} name="Fed Funds %" strokeWidth={2} />
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
              <Card style={{ backgroundColor: COLORS.bgSurface, borderColor: COLORS.border }}>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2 font-mono" style={{ color: COLORS.textPrimary }}>
                    <Zap className="w-5 h-5" style={{ color: COLORS.accentCyan }} />
                    GLOBAL FEATURE IMPORTANCE (SHAP)
                  </CardTitle>
                  <CardDescription style={{ color: COLORS.textSecondary }}>SR 11-7 Model Risk Management Compliance</CardDescription>
                </CardHeader>
                <CardContent>
                  <Button 
                    data-testid="compute-shap-btn"
                    onClick={fetchShapExplanation}
                    disabled={loading.shap || !modelStatus?.all_ready}
                    className="w-full font-mono mb-4"
                    style={{ backgroundColor: COLORS.accentCyan, color: COLORS.bgPrimary }}
                  >
                    {loading.shap ? <RefreshCw className="w-4 h-4 mr-2 animate-spin" /> : <Zap className="w-4 h-4 mr-2" />}
                    COMPUTE SHAP VALUES
                  </Button>
                  
                  {shapExplanation && (
                    <div className="space-y-3">
                      {Object.entries(shapExplanation.global_importance)
                        .sort(([,a], [,b]) => b - a)
                        .map(([feature, importance]) => {
                          const pct = (importance / Object.values(shapExplanation.global_importance).reduce((a,b) => a+b, 0)) * 100;
                          return (
                            <div key={feature}>
                              <div className="flex justify-between text-sm font-mono mb-1">
                                <span style={{ color: COLORS.textPrimary }}>{feature.toUpperCase().replace(/_/g, ' ')}</span>
                                <span style={{ color: COLORS.accentCyan }}>{pct.toFixed(1)}%</span>
                              </div>
                              <div className="h-2 rounded" style={{ backgroundColor: COLORS.bgElevated }}>
                                <div 
                                  className="h-full rounded transition-all duration-500"
                                  style={{ width: `${pct}%`, backgroundColor: COLORS.accentCyan }}
                                />
                              </div>
                            </div>
                          );
                        })}
                    </div>
                  )}
                </CardContent>
              </Card>

              <Card style={{ backgroundColor: COLORS.bgSurface, borderColor: COLORS.border }}>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2 font-mono" style={{ color: COLORS.textPrimary }}>
                    <Brain className="w-5 h-5" style={{ color: COLORS.accentGreen }} />
                    AI-POWERED ANALYSIS (GEMINI)
                  </CardTitle>
                  <CardDescription style={{ color: COLORS.textSecondary }}>Natural language interpretation</CardDescription>
                </CardHeader>
                <CardContent>
                  <Button 
                    data-testid="generate-ai-explanation-btn"
                    onClick={fetchAiExplanation}
                    disabled={loading.ai || !modelStatus?.all_ready}
                    className="w-full font-mono mb-4"
                    style={{ backgroundColor: COLORS.accentGreen, color: COLORS.bgPrimary }}
                  >
                    {loading.ai ? <RefreshCw className="w-4 h-4 mr-2 animate-spin" /> : <Brain className="w-4 h-4 mr-2" />}
                    GENERATE AI EXPLANATION
                  </Button>
                  
                  {aiExplanation && (
                    <div className="p-4 rounded" style={{ backgroundColor: COLORS.bgElevated, borderLeft: `3px solid ${COLORS.accentGreen}` }}>
                      <p className="leading-relaxed whitespace-pre-wrap" style={{ color: COLORS.textPrimary }}>
                        {aiExplanation}
                      </p>
                    </div>
                  )}
                </CardContent>
              </Card>
            </div>

            <Alert style={{ backgroundColor: COLORS.bgSurface, borderColor: COLORS.border }}>
              <Shield className="h-5 w-5" style={{ color: COLORS.accentCyan }} />
              <AlertTitle className="font-mono" style={{ color: COLORS.textPrimary }}>SR 11-7 MODEL RISK MANAGEMENT</AlertTitle>
              <AlertDescription style={{ color: COLORS.textSecondary }}>
                This analysis complies with Federal Reserve SR 11-7 guidelines for model risk management, 
                providing transparent feature attribution and scenario-specific explanations for regulatory review.
              </AlertDescription>
            </Alert>
          </TabsContent>

          {/* Reports Tab */}
          <TabsContent data-testid="reports-tab-content" value="reports" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              <Card style={{ backgroundColor: COLORS.bgSurface, borderColor: COLORS.border }}>
                <CardHeader>
                  <CardTitle className="font-mono" style={{ color: COLORS.textPrimary }}>CET1 TRAJECTORY</CardTitle>
                </CardHeader>
                <CardContent>
                  <Button 
                    data-testid="download-trajectory-btn"
                    disabled={!forecastResults}
                    className="w-full font-mono"
                    style={{ backgroundColor: COLORS.accentCyan, color: COLORS.bgPrimary }}
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
                    DOWNLOAD CSV
                  </Button>
                </CardContent>
              </Card>

              <Card style={{ backgroundColor: COLORS.bgSurface, borderColor: COLORS.border }}>
                <CardHeader>
                  <CardTitle className="font-mono" style={{ color: COLORS.textPrimary }}>EXECUTIVE SUMMARY</CardTitle>
                </CardHeader>
                <CardContent>
                  <Button 
                    data-testid="download-summary-btn"
                    disabled={!forecastResults}
                    className="w-full font-mono"
                    style={{ backgroundColor: COLORS.accentGreen, color: COLORS.bgPrimary }}
                  >
                    <Download className="w-4 h-4 mr-2" />
                    DOWNLOAD PDF
                  </Button>
                </CardContent>
              </Card>

              <Card style={{ backgroundColor: COLORS.bgSurface, borderColor: COLORS.border }}>
                <CardHeader>
                  <CardTitle className="font-mono" style={{ color: COLORS.textPrimary }}>SHAP ANALYSIS</CardTitle>
                </CardHeader>
                <CardContent>
                  <Button 
                    data-testid="download-shap-btn"
                    disabled={!shapExplanation}
                    className="w-full font-mono"
                    style={{ backgroundColor: COLORS.accentYellow, color: COLORS.bgPrimary }}
                  >
                    <Download className="w-4 h-4 mr-2" />
                    DOWNLOAD REPORT
                  </Button>
                </CardContent>
              </Card>
            </div>

            {forecastResults?.trajectory && (
              <Card style={{ backgroundColor: COLORS.bgSurface, borderColor: COLORS.border }}>
                <CardHeader>
                  <CardTitle className="font-mono" style={{ color: COLORS.textPrimary }}>BANK-LEVEL RESULTS SUMMARY</CardTitle>
                </CardHeader>
                <CardContent>
                  <ScrollArea className="h-96">
                    <table className="w-full text-sm font-mono">
                      <thead style={{ backgroundColor: COLORS.bgElevated }}>
                        <tr>
                          <th className="text-left p-3" style={{ color: COLORS.textSecondary }}>BANK</th>
                          <th className="text-right p-3" style={{ color: COLORS.textSecondary }}>INITIAL CET1</th>
                          <th className="text-right p-3" style={{ color: COLORS.textSecondary }}>MIN CET1</th>
                          <th className="text-right p-3" style={{ color: COLORS.textSecondary }}>FINAL CET1</th>
                          <th className="text-right p-3" style={{ color: COLORS.textSecondary }}>TOTAL LOSSES</th>
                          <th className="text-center p-3" style={{ color: COLORS.textSecondary }}>STATUS</th>
                        </tr>
                      </thead>
                      <tbody>
                        {[...new Set(forecastResults.trajectory.map(t => t.bank_id))].map(bankId => {
                          const bankDataRow = forecastResults.trajectory.filter(t => t.bank_id === bankId);
                          const initial = bankDataRow.find(t => t.quarter === 1);
                          const final = bankDataRow.find(t => t.quarter === 9);
                          const minCet1 = Math.min(...bankDataRow.map(t => t.cet1_ratio));
                          const totalLosses = bankDataRow.reduce((sum, t) => sum + t.credit_losses, 0);
                          const hasBreach = bankDataRow.some(t => t.breach);
                          
                          return (
                            <tr key={bankId} style={{ borderBottom: `1px solid ${COLORS.border}` }}>
                              <td className="p-3" style={{ color: COLORS.textPrimary }}>{bankDataRow[0]?.bank_name || bankId}</td>
                              <td className="p-3 text-right" style={{ color: COLORS.textPrimary }}>
                                {((initial?.cet1_ratio || 0) * 100).toFixed(2)}%
                              </td>
                              <td className="p-3 text-right" style={{ color: minCet1 < 0.045 ? COLORS.accentRed : minCet1 < 0.07 ? COLORS.accentYellow : COLORS.accentGreen }}>
                                {(minCet1 * 100).toFixed(2)}%
                              </td>
                              <td className="p-3 text-right" style={{ color: COLORS.textPrimary }}>
                                {((final?.cet1_ratio || 0) * 100).toFixed(2)}%
                              </td>
                              <td className="p-3 text-right" style={{ color: COLORS.accentRed }}>
                                ${(totalLosses / 1e9).toFixed(2)}B
                              </td>
                              <td className="p-3 text-center">
                                <Badge 
                                  className="font-mono text-xs px-2 py-1"
                                  style={{ 
                                    backgroundColor: hasBreach ? 'rgba(255, 49, 49, 0.15)' : 'rgba(0, 255, 65, 0.15)',
                                    color: hasBreach ? COLORS.accentRed : COLORS.accentGreen,
                                    border: `1px solid ${hasBreach ? 'rgba(255, 49, 49, 0.3)' : 'rgba(0, 255, 65, 0.3)'}`
                                  }}
                                >
                                  {hasBreach ? 'BREACH' : 'SAFE'}
                                </Badge>
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

      <footer className="py-6 mt-12" style={{ borderTop: `1px solid ${COLORS.border}` }}>
        <div className="container mx-auto px-6 text-center font-mono text-sm" style={{ color: COLORS.textSecondary }}>
          <p>REG-EXPLAIN | CCAR FRAMEWORK 2026 | PPNR STRESS-TESTING VIA HYBRID LSTM-XGBOOST</p>
          <p className="mt-1" style={{ color: COLORS.textMuted }}>COMPLIANT WITH SR 11-7 MODEL RISK MANAGEMENT GUIDELINES</p>
        </div>
      </footer>
    </div>
  );
}

export default App;
