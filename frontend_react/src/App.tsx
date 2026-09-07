import React, { useState, useEffect, useCallback } from 'react';
import ReactFlow, { Background, Controls, MarkerType, useNodesState, useEdgesState, Handle, Position } from 'reactflow';
import 'reactflow/dist/style.css';
import { Play, Upload, Activity, Database, Terminal, Image as ImageIcon, FileText, Download, X } from 'lucide-react';

const backendUrl = "";

const CustomNode = ({ data, selected }) => (  <div className={`cursor-pointer px-4 py-2 rounded-lg border-2 transition-all duration-300 ${
    data.status === 'running' ? 'bg-primary/20 border-primary shadow-[0_0_15px_rgba(59,130,246,0.5)]' :
    data.status === 'complete' ? 'bg-green-500/20 border-green-500 shadow-[0_0_10px_rgba(34,197,94,0.3)]' :
    data.status === 'error' ? 'bg-red-500/20 border-red-500 shadow-[0_0_10px_rgba(239,68,68,0.3)]' :
    'bg-panel border-white/10'
  } ${selected ? 'ring-2 ring-accent' : ''}`}>
    <Handle type="target" position={Position.Top} className="opacity-0" />
    <div className="flex items-center gap-2">
      {data.status === 'running' && <Activity size={16} className="animate-pulse text-primary" />}
      <span className="font-semibold">{data.label}</span>
    </div>
    <Handle type="source" position={Position.Bottom} className="opacity-0" />
  </div>
);

const nodeTypes = { custom: CustomNode };

const initialNodes = [
  { id: 'vision', type: 'custom', data: { label: 'Vision Agent', status: 'idle' }, position: { x: 100, y: 50 } },
  { id: 'tumor_classification', type: 'custom', data: { label: 'Tumor Classification', status: 'idle' }, position: { x: 350, y: 150 } },
  { id: 'localization', type: 'custom', data: { label: 'Localization', status: 'idle' }, position: { x: 100, y: 250 } },
  { id: 'emergency', type: 'custom', data: { label: 'Emergency', status: 'idle' }, position: { x: 350, y: 350 } },
  { id: 'surgical', type: 'custom', data: { label: 'Surgical Planning', status: 'idle' }, position: { x: 100, y: 450 } },
  { id: 'prognostic', type: 'custom', data: { label: 'Prognostic', status: 'idle' }, position: { x: 350, y: 550 } },
  { id: 'clinical_trials', type: 'custom', data: { label: 'Clinical Trials', status: 'idle' }, position: { x: 100, y: 650 } },
  { id: 'neuro_oncologist', type: 'custom', data: { label: 'Neuro-Oncologist', status: 'idle' }, position: { x: 350, y: 750 } },
  { id: 'explainability', type: 'custom', data: { label: 'Explainability', status: 'idle' }, position: { x: 225, y: 850 } },
];

const initialEdges = initialNodes.slice(0, -1).map((node, i) => ({
  id: `e-${node.id}-${initialNodes[i+1].id}`,
  source: node.id,
  target: initialNodes[i+1].id,
  animated: true,
  style: { stroke: '#fff', opacity: 0.2 },
  markerEnd: { type: MarkerType.ArrowClosed, color: '#fff' }
}));

export default function App() {
  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);
  const [selectedAgent, setSelectedAgent] = useState(null);
  const [agentData, setAgentData] = useState({});
  const [file, setFile] = useState(null);
  const [patientData, setPatientData] = useState({ age: 50, sex: 'M' });
  const [isRunning, setIsRunning] = useState(false);
  const [hasRun, setHasRun] = useState(false);
  const [fullScreenImage, setFullScreenImage] = useState(null);
  const [showReport, setShowReport] = useState(false);

  const generateReportText = () => {
    let report = `NEURO AGENT - MEDICAL GRADED REPORT\n`;
    report += `=====================================\n`;
    report += `Date: ${new Date().toLocaleDateString()}\n`;
    report += `Patient Age: ${patientData.age}\n`;
    report += `Patient Sex: ${patientData.sex}\n\n`;
    
    const explainData = agentData['explainability'];
    
    report += `EXPLAINABILITY SUMMARY:\n`;
    report += `-----------------------\n`;
    
    if (explainData && explainData.output_data && explainData.output_data.explanation_summary) {
      report += `${explainData.output_data.explanation_summary}\n`;
    } else {
      report += `No summary available from the Explainability agent.\n`;
    }
    
    report += `\n=====================================\n`;
    report += `End of Report\n`;
    return report;
  };

  const downloadReport = () => {
    const reportText = generateReportText();
    const blob = new Blob([reportText], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `NeuroAgent_Report_${new Date().getTime()}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const onNodeClick = (event, node) => {
    setSelectedAgent(node.id);
  };

  const resetAnalysis = () => {
    setFile(null);
    setHasRun(false);
    setIsRunning(false);
    setAgentData({});
    setSelectedAgent(null);
    setNodes(nds => nds.map(n => ({ ...n, data: { ...n.data, status: 'idle' } })));
    setEdges(eds => eds.map(e => ({ ...e, style: { stroke: '#fff', opacity: 0.2 }, animated: true })));
  };

  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === 'Escape') setFullScreenImage(null);
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  const handleRun = async () => {
    if (!file) {
      alert("Please upload a file first");
      return;
    }
    
    setIsRunning(true);
    setHasRun(true);
    setAgentData({});
    setNodes(nds => nds.map(n => ({ ...n, data: { ...n.data, status: 'idle' } })));
    setEdges(eds => eds.map(e => ({ ...e, style: { stroke: '#fff', opacity: 0.2 } })));
    
    const formData = new FormData();
    formData.append("scan_file", file);
    formData.append("patient_json", JSON.stringify({
      age: patientData.age,
      sex: patientData.sex,
      symptoms: [], medical_history: [], medications: [], referring_notes: "", scan_modality: "FLAIR"
    }));

    try {
      const response = await fetch(`${backendUrl}/api/analyze/stream`, {
        method: "POST",
        body: formData,
      });

      if (!response.body) throw new Error("No response body");

      const reader = response.body.getReader();
      const decoder = new TextDecoder("utf-8");

      let buffer = "";
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        
        buffer += decoder.decode(value, { stream: true });
        
        const lines = buffer.split('\n');
        buffer = lines.pop() || "";
        
        for (const line of lines) {
          if (line.startsWith("data: ")) {
            const rawData = line.substring(6);
            try {
              const data = JSON.parse(rawData);
              console.log("Event:", data);
              const { agent_name, status } = data;
              
              setAgentData(prev => ({
                ...prev,
                [agent_name]: data
              }));
              
              if (status === 'running') setSelectedAgent(agent_name);
              
              setNodes(nds => nds.map(n => {
                if (n.id === agent_name) {
                  return { ...n, data: { ...n.data, status } };
                }
                return n;
              }));

              if (status === 'complete') {
                setEdges(eds => eds.map(e => {
                  if (e.source === agent_name) {
                    return { ...e, style: { stroke: '#3b82f6', strokeWidth: 2, opacity: 1 }, animated: false };
                  }
                  return e;
                }));
              }
              
              if (data.is_final) {
                setIsRunning(false);
              }
            } catch(e) {
              console.error("JSON parse error:", e);
            }
          }
        }
      }
    } catch (e) {
      console.error(e);
      setIsRunning(false);
    }
  };

  const currentData = selectedAgent ? agentData[selectedAgent] : null;

  return (
    <div className="flex h-screen w-screen bg-background text-slate-200 p-4 gap-4 overflow-hidden font-sans">

      <div className="w-1/4 min-w-[300px] glass-panel rounded-2xl p-6 flex flex-col gap-6 shadow-2xl relative">
        <div>
          <h2 className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-accent to-primary mb-1">Neuro Agent</h2>
          <p className="text-sm text-slate-400">Autonomous Diagnostic Pipeline</p>
        </div>
        
        {hasRun && file ? (
          <div className="flex flex-col gap-4 flex-1">
            <div className="relative rounded-xl overflow-hidden border border-slate-700 bg-black flex-1 shadow-[0_0_30px_rgba(0,240,255,0.15)] flex items-center justify-center">
              <img src={URL.createObjectURL(file)} alt="MRI Scan" className="w-full h-full object-cover opacity-80" />
              {/* Scanning laser animation overlay */}
              {isRunning && (
                <>
                  <div className="absolute left-0 w-full h-[2px] bg-accent shadow-[0_0_15px_3px_rgba(0,240,255,0.7)] animate-scan z-20"></div>
                  <div className="absolute left-0 w-full h-[150px] bg-gradient-to-t from-accent/20 to-transparent animate-scan z-10 -mt-[150px]"></div>
                </>
              )}
              
              <div className="absolute top-3 left-3 bg-black/60 backdrop-blur-sm border border-accent/30 rounded-md px-2 py-1 flex items-center gap-2">
                <div className={`w-2 h-2 rounded-full ${isRunning ? 'bg-accent animate-pulse' : 'bg-green-500'}`}></div>
                <span className="text-[10px] uppercase font-bold tracking-widest text-accent">
                  {isRunning ? 'Scanning...' : 'Analysis Complete'}
                </span>
              </div>
            </div>
            
            <div className="bg-slate-900/50 rounded-xl p-4 border border-slate-800">
              <div className="flex justify-between text-xs mb-2">
                <span className="text-slate-500">PATIENT AGE</span>
                <span className="font-mono text-slate-300">{patientData.age}</span>
              </div>
              <div className="flex justify-between text-xs">
                <span className="text-slate-500">PATIENT SEX</span>
                <span className="font-mono text-slate-300">{patientData.sex}</span>
              </div>
            </div>

            {!isRunning && (
              <div className="flex flex-col gap-2 mt-auto">
                {agentData['explainability']?.status === 'complete' && (
                  <button 
                    onClick={() => setShowReport(true)}
                    className="bg-primary hover:bg-primary/80 text-white font-semibold py-3 px-4 rounded-xl flex items-center justify-center gap-2 transition-all shadow-[0_0_20px_rgba(59,130,246,0.3)] hover:shadow-[0_0_30px_rgba(59,130,246,0.5)]"
                  >
                    <FileText size={18} /> View Final Report
                  </button>
                )}
                <button 
                  onClick={resetAnalysis}
                  className="bg-slate-800 hover:bg-slate-700 border border-slate-700 text-white font-semibold py-3 px-4 rounded-xl flex items-center justify-center gap-2 transition-all shadow-lg"
                >
                  <Upload size={18} /> Test Another MRI
                </button>
              </div>
            )}
          </div>
        ) : (
          <>
            <div className="flex flex-col gap-4">
              <label className="border-2 border-dashed border-slate-700 bg-slate-900/50 rounded-xl p-8 flex flex-col items-center justify-center cursor-pointer hover:border-accent/50 hover:bg-slate-900 transition-all group">
                <Upload size={32} className="mb-2 text-slate-500 group-hover:text-accent transition-colors" />
                <p className="text-sm font-medium text-slate-400 group-hover:text-slate-300">
                  {file ? file.name : "Upload MRI Scan"}
                </p>
                <input type="file" className="hidden" onChange={e => setFile(e.target.files[0])} />
              </label>
              
              <div className="flex gap-4">
                <div className="flex-1">
                  <label className="text-xs text-slate-500 uppercase tracking-wider font-semibold mb-1 block">Age</label>
                  <input type="number" value={patientData.age} onChange={e => setPatientData({...patientData, age: parseInt(e.target.value)})} className="w-full bg-slate-900/50 border border-slate-800 rounded-lg p-2 text-slate-200 focus:outline-none focus:border-accent" />
                </div>
                <div className="flex-1">
                  <label className="text-xs text-slate-500 uppercase tracking-wider font-semibold mb-1 block">Sex</label>
                  <select value={patientData.sex} onChange={e => setPatientData({...patientData, sex: e.target.value})} className="w-full bg-slate-900/50 border border-slate-800 rounded-lg p-2 text-slate-200 focus:outline-none focus:border-accent">
                    <option>M</option><option>F</option><option>Other</option>
                  </select>
                </div>
              </div>
            </div>

            <button 
              onClick={handleRun}
              disabled={isRunning || !file}
              className="mt-auto bg-primary hover:bg-primary/80 disabled:opacity-50 disabled:cursor-not-allowed text-white font-semibold py-3 px-4 rounded-xl flex items-center justify-center gap-2 transition-all shadow-[0_0_20px_rgba(59,130,246,0.3)] hover:shadow-[0_0_30px_rgba(59,130,246,0.5)]"
            >
              {isRunning ? <Activity size={20} className="animate-spin" /> : <Play size={20} />} 
              {isRunning ? 'Processing...' : 'Run Analysis'}
            </button>
          </>
        )}
      </div>


      <div className="flex-1 glass-panel rounded-2xl overflow-hidden relative shadow-2xl">
        <div className="absolute top-4 left-4 z-10 bg-slate-900/80 backdrop-blur px-3 py-1.5 rounded-full border border-slate-800 text-xs font-medium text-slate-300 flex items-center gap-2">
          <div className={`w-2 h-2 rounded-full ${isRunning ? 'bg-primary animate-pulse' : 'bg-slate-500'}`}></div>
          Pipeline Status
        </div>
        <ReactFlow 
          nodes={nodes} 
          edges={edges} 
          nodeTypes={nodeTypes}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onNodeClick={onNodeClick}
          fitView
          attributionPosition="bottom-left"
        >
          <Background color="#fff" gap={20} size={1} opacity={0.05} />
          <Controls className="bg-slate-900 border-slate-800 fill-white" />
        </ReactFlow>
      </div>


      <div className="w-[520px] min-w-[400px] max-w-[550px] glass-panel rounded-2xl flex flex-col shadow-2xl overflow-hidden">
        <div className="p-4 bg-slate-900/60 backdrop-blur flex justify-between items-center">
          <h2 className="text-lg font-bold flex items-center gap-2">
            <Terminal size={18} className="text-accent" />
            Agent Insights
          </h2>
          {selectedAgent && (
             <span className="text-xs bg-slate-800 text-slate-300 px-2 py-1 rounded-md uppercase tracking-wide">
               {selectedAgent.replace('_', ' ')}
             </span>
          )}
        </div>
        
        <div className="flex-1 overflow-y-auto p-4 flex flex-col gap-5 custom-scrollbar">
          {!selectedAgent ? (
            <div className="h-full flex flex-col items-center justify-center text-slate-500 gap-3">
              <Database size={40} className="opacity-20" />
              <p>Select a node to view its execution trace.</p>
            </div>
          ) : !currentData ? (
            <div className="h-full flex flex-col items-center justify-center text-slate-500 gap-3">
              <Activity size={32} className="opacity-20 animate-pulse" />
              <p>Awaiting execution...</p>
            </div>
          ) : (() => {
            const getImageSrc = () => {
              if (currentData?.image_b64) {
                if (currentData.image_b64.startsWith('data:') || currentData.image_b64.startsWith('http')) {
                  return currentData.image_b64;
                }
                const isJpeg = currentData.image_b64.startsWith('/9j/');
                return `data:image/${isJpeg ? 'jpeg' : 'png'};base64,${currentData.image_b64}`;
              }
              const rawPath = currentData?.output_data?.segmentation_mask_path || 
                              currentData?.output_data?.midline_shift_image_path || 
                              currentData?.output_data?.gradcam_heatmap_path;
              if (rawPath) {
                const cleanPath = rawPath.replace(/\\/g, '/');
                return `${backendUrl}/${cleanPath}`;
              }
              return null;
            };
            const activeImageSrc = getImageSrc();

            return (
              <>
                {/* Clinical status banner */}
                {currentData.output_data?.tumor_detected === false && (
                  <div 
                    className={`bg-emerald-950/40 border border-emerald-500/40 rounded-xl p-3 flex items-center justify-between transition-all ${activeImageSrc ? 'cursor-pointer hover:border-emerald-400/80 hover:bg-emerald-950/60' : ''}`}
                    onClick={() => activeImageSrc && setFullScreenImage(activeImageSrc)}
                  >
                    <div className="flex items-center gap-3">
                      <div className="w-3.5 h-3.5 rounded-full bg-emerald-400 animate-pulse shadow-[0_0_10px_rgba(52,211,153,0.8)]"></div>
                      <div>
                        <div className="text-emerald-400 font-bold text-sm flex items-center gap-2">
                          No Tumor Detected (Normal Study)
                          {activeImageSrc && <span className="text-[11px] font-normal text-emerald-300/80 bg-emerald-900/40 border border-emerald-800/60 px-2 py-0.5 rounded-full">🔍 View scan</span>}
                        </div>
                        <div className="text-emerald-300/70 text-xs">YOLO and anatomical screening confirmed healthy brain parenchyma</div>
                      </div>
                    </div>
                  </div>
                )}
                {currentData.output_data?.tumor_detected === true && (
                  <div 
                    className={`bg-red-950/40 border border-red-500/40 rounded-xl p-3 flex items-center justify-between transition-all ${activeImageSrc ? 'cursor-pointer hover:border-red-400/80 hover:bg-red-950/60 shadow-lg' : ''}`}
                    onClick={() => activeImageSrc && setFullScreenImage(activeImageSrc)}
                    title={activeImageSrc ? "Click to view full screen MRI scan" : undefined}
                  >
                    <div className="flex items-center gap-3">
                      <div className="w-3.5 h-3.5 rounded-full bg-red-500 animate-pulse shadow-[0_0_10px_rgba(239,68,68,0.8)]"></div>
                      <div>
                        <div className="text-red-400 font-bold text-sm flex items-center gap-2">
                          Tumor Mass Identified
                          {activeImageSrc && <span className="text-[11px] font-normal text-red-300/80 bg-red-900/40 border border-red-800/60 px-2 py-0.5 rounded-full flex items-center gap-1">🔍 Click to enlarge</span>}
                        </div>
                        <div className="text-red-300/70 text-xs">
                          Area: {currentData.output_data.tumor_area_cm2} cm² | Vol: {currentData.output_data.tumor_volume_cm3} cm³
                        </div>
                      </div>
                    </div>
                    {currentData.output_data.mri_sequence && (
                      <span className="text-xs bg-red-900/60 text-red-200 border border-red-700/50 px-2.5 py-1 rounded font-mono font-semibold">
                        {currentData.output_data.mri_sequence}
                      </span>
                    )}
                  </div>
                )}

                {activeImageSrc && (
                  <div 
                    className="rounded-xl overflow-hidden relative group cursor-pointer hover:border-accent/50 transition-all shadow-lg bg-black border border-slate-700/50"
                    onClick={() => setFullScreenImage(activeImageSrc)}
                  >
                    <div className="absolute inset-0 bg-gradient-to-t from-black/80 to-transparent z-10 opacity-0 group-hover:opacity-100 transition-opacity flex items-end justify-center pb-4">
                      <span className="text-sm font-medium text-white flex items-center gap-2 bg-black/60 px-3 py-1.5 rounded-lg backdrop-blur-md">
                        <ImageIcon size={16}/> Click to enlarge
                      </span>
                    </div>
                    <img 
                      src={activeImageSrc} 
                      alt="Agent Output" 
                      className="w-full block"
                      style={{ objectFit: 'contain', minHeight: '280px', maxHeight: '420px', display: 'block' }}
                    />
                  </div>
                )}

                <div>
                  <h3 className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-2">Agent Task / Input</h3>
                  <div className="bg-slate-900/80 rounded-lg p-3 font-mono text-xs text-blue-300 border border-slate-800 overflow-x-auto">
                    <pre>{JSON.stringify(currentData.input_data, null, 2)}</pre>
                  </div>
                </div>

                <div>
                  <h3 className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-2">Execution Trace</h3>
                  <div className="bg-black/60 rounded-lg p-3 font-mono text-xs text-emerald-400 border border-slate-800">
                    {currentData.processing_logs?.map((log, i) => (
                      <div key={i} className="mb-1 opacity-90 animate-[fadeIn_0.3s_ease-out]">
                        <span className="text-slate-600 mr-2">[{new Date().toISOString().split('T')[1].slice(0,12)}]</span>
                        {log}
                      </div>
                    ))}
                    {currentData.status === 'running' && (
                      <div className="animate-pulse">_</div>
                    )}
                  </div>
                </div>

                {Object.keys(currentData.output_data || {}).length > 0 && (
                  <div>
                    <h3 className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-2">Structured Output</h3>
                    <div className="bg-slate-900/80 rounded-lg p-3 font-mono text-xs text-amber-300 border border-slate-800 overflow-x-auto">
                      <pre>{JSON.stringify(currentData.output_data, null, 2)}</pre>
                    </div>
                  </div>
                )}
              </>
            );
          })()}
        </div>
      </div>
      
      {fullScreenImage && (
        <div 
          className="fixed inset-0 z-50 bg-black/90 backdrop-blur-sm flex items-center justify-center p-4 cursor-pointer"
          onClick={() => setFullScreenImage(null)}
        >
          <img 
            src={fullScreenImage.startsWith('data:') || fullScreenImage.startsWith('http') ? fullScreenImage : `data:image/jpeg;base64,${fullScreenImage}`} 
            alt="Full Screen Output" 
            style={{ width: '90vw', height: '90vh', objectFit: 'contain' }}
            className="rounded-xl shadow-[0_0_50px_rgba(0,240,255,0.2)] border border-white/10"
          />

          <div className="absolute top-6 right-6 text-white/50 hover:text-white transition-colors">
            <span className="bg-black/50 px-4 py-2 rounded-lg backdrop-blur-md border border-white/10 text-sm">Esc or Click to close</span>
          </div>
        </div>
      )}

      {showReport && (
        <div className="fixed inset-0 z-50 bg-black/80 backdrop-blur-sm flex items-center justify-center p-8">
          <div className="bg-slate-900 border border-slate-700 rounded-2xl shadow-2xl w-full max-w-3xl max-h-full flex flex-col overflow-hidden">
            <div className="p-4 border-b border-slate-800 bg-slate-900/50 backdrop-blur flex justify-between items-center">
              <h2 className="text-xl font-bold flex items-center gap-2 text-white">
                <FileText size={20} className="text-accent" />
                Medical Graded Report
              </h2>
              <button 
                onClick={() => setShowReport(false)}
                className="text-slate-400 hover:text-white transition-colors"
              >
                <X size={24} />
              </button>
            </div>
            
            <div className="flex-1 overflow-y-auto p-6 bg-black/30 font-mono text-sm text-slate-300 custom-scrollbar">
              <pre className="whitespace-pre-wrap">{generateReportText()}</pre>
            </div>
            
            <div className="p-4 border-t border-slate-800 bg-slate-900 flex justify-end gap-3">
              <button 
                onClick={() => setShowReport(false)}
                className="px-4 py-2 rounded-lg font-semibold text-slate-300 hover:bg-slate-800 transition-colors"
              >
                Close
              </button>
              <button 
                onClick={downloadReport}
                className="px-4 py-2 rounded-lg font-semibold bg-primary hover:bg-primary/80 text-white flex items-center gap-2 transition-colors shadow-lg"
              >
                <Download size={18} /> Download Text Report
              </button>
            </div>
          </div>
        </div>
      )}

      <style dangerouslySetInnerHTML={{__html: `
        .custom-scrollbar { /* Fixed the valid UTF8 but bad template issue */ }
        .custom-scrollbar::-webkit-scrollbar { width: 6px; }
        .custom-scrollbar::-webkit-scrollbar-track { background: rgba(0,0,0,0.2); }
        .custom-scrollbar::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 10px; }
        .custom-scrollbar::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.2); }
      `}} />
    </div>
  );
}

