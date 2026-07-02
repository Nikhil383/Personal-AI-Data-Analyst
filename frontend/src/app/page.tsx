"use client";

import React, { useState, useEffect, useRef } from "react";
import * as echarts from "echarts";
import { Play, Database, ShieldAlert, Sparkles, BarChart3, LineChart, FileText, Send, Loader2 } from "lucide-react";

interface QueryResponse {
  query: string;
  intent: string;
  sql_query?: string;
  sql_valid?: boolean;
  sql_error?: string;
  query_results?: Array<Record<string, any>>;
  visualization?: string;
  insights?: string;
}

export default function Home() {
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [history, setHistory] = useState<QueryResponse[]>([]);
  const [schema, setSchema] = useState<{ tables: Record<string, string[]> }>({ tables: {} });
  const [currentResponse, setCurrentResponse] = useState<QueryResponse | null>(null);
  const [initStatus, setInitStatus] = useState<string>("");
  const chartRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // Fetch schema information
    fetch("http://localhost:8000/api/schema")
      .then((res) => res.json())
      .then((data) => setSchema(data))
      .catch((err) => console.error("Error fetching schema:", err));
  }, []);

  // Initialize DB helper
  const handleInitDB = async () => {
    setInitStatus("Initializing...");
    try {
      const res = await fetch("http://localhost:8000/api/db/init", { method: "POST" });
      const data = await res.json();
      setInitStatus(data.message || "DB Initialized!");
    } catch (err) {
      setInitStatus("Failed to initialize database.");
    }
  };

  const handleQuery = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;

    setLoading(true);
    try {
      const res = await fetch("http://localhost:8000/api/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query }),
      });
      const data: QueryResponse = await res.json();
      setCurrentResponse(data);
      setHistory((prev) => [data, ...prev]);

      // Render chart if visualization is recommended
      if (data.visualization) {
        setTimeout(() => {
          renderChart(data);
        }, 100);
      }
    } catch (err) {
      console.error("Query execution failed:", err);
    } finally {
      setLoading(false);
      setQuery("");
    }
  };

  const renderChart = (data: QueryResponse) => {
    if (!chartRef.current) return;
    echarts.dispose(chartRef.current);
    const myChart = echarts.init(chartRef.current);

    try {
      // Try to parse the LLM recommendation or fallback to default
      let option: any = {};
      const match = data.visualization?.match(/\{[\s\S]*\}/);
      if (match) {
        option = JSON.parse(match[0]);
      } else {
        // Fallback option
        option = {
          title: { text: "Query Output Visualization", textStyle: { color: "#ffffff" } },
          tooltip: {},
          xAxis: { data: data.query_results?.map((r, idx) => `Item ${idx + 1}`) },
          yAxis: {},
          series: [{ type: "bar", data: data.query_results?.map((r) => Object.values(r)[0]) }],
        };
      }

      // Ensure style compatibility with dark theme
      option.backgroundColor = "transparent";
      if (option.title) option.title.textStyle = { color: "#f3f4f6" };
      if (option.legend) option.legend.textStyle = { color: "#9ca3af" };
      if (option.xAxis) option.xAxis.axisLabel = { color: "#9ca3af" };
      if (option.yAxis) option.yAxis.axisLabel = { color: "#9ca3af" };

      myChart.setOption(option);
    } catch (err) {
      console.error("ECharts rendering error:", err);
    }
  };

  return (
    <div style={{ display: "flex", height: "100vh", overflow: "hidden" }}>
      {/* Sidebar - Schema details */}
      <div style={{ width: "300px", borderRight: "1px solid #1f2937", padding: "20px", display: "flex", flexDirection: "column", gap: "20px", backgroundColor: "#0f172a" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
          <Database size={24} color="#3b82f6" />
          <h2 style={{ fontSize: "1.2rem", fontWeight: "bold", margin: 0 }}>Data Schema</h2>
        </div>
        
        <button onClick={handleInitDB} style={{ width: "100%", padding: "10px", backgroundColor: "#2563eb", color: "#ffffff", border: "none", borderRadius: "6px", cursor: "pointer", fontWeight: "bold" }}>
          Seed Mock Database
        </button>
        {initStatus && <div style={{ fontSize: "0.85rem", color: "#9ca3af" }}>{initStatus}</div>}

        <div style={{ display: "flex", flexDirection: "column", gap: "15px", overflowY: "auto", flex: 1 }}>
          {Object.entries(schema.tables).map(([table, cols]) => (
            <div key={table} style={{ backgroundColor: "#1e293b", padding: "12px", borderRadius: "8px" }}>
              <div style={{ fontWeight: "bold", color: "#60a5fa", marginBottom: "8px" }}>{table}</div>
              <div style={{ display: "flex", flexWrap: "wrap", gap: "6px" }}>
                {cols.map((col) => (
                  <span key={col} style={{ fontSize: "0.75rem", backgroundColor: "#334155", color: "#cbd5e1", padding: "2px 6px", borderRadius: "4px" }}>
                    {col}
                  </span>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Main Workspace */}
      <div style={{ flex: 1, display: "flex", flexDirection: "column", height: "100%", backgroundColor: "#0b0f19" }}>
        {/* Header */}
        <div style={{ borderBottom: "1px solid #1f2937", padding: "15px 30px", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <h1 style={{ fontSize: "1.3rem", fontWeight: "bold", margin: 0, display: "flex", alignItems: "center", gap: "8px" }}>
            <Sparkles color="#3b82f6" /> Enterprise AI Data Analyst
          </h1>
        </div>

        {/* Dashboard Panels */}
        <div style={{ flex: 1, display: "flex", overflow: "hidden" }}>
          {/* Left - Workspace / Query Output */}
          <div style={{ flex: 3, display: "flex", flexDirection: "column", padding: "20px", gap: "20px", overflowY: "auto" }}>
            {currentResponse ? (
              <>
                {/* SQL & Security Validation */}
                {currentResponse.sql_query && (
                  <div style={{ backgroundColor: "#1e293b", padding: "20px", borderRadius: "12px", border: "1px solid #334155" }}>
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "10px" }}>
                      <span style={{ fontWeight: "bold", fontSize: "0.95rem" }}>SQL Query</span>
                      {currentResponse.sql_valid ? (
                        <span style={{ fontSize: "0.75rem", color: "#10b981", backgroundColor: "rgba(16, 185, 129, 0.1)", padding: "4px 8px", borderRadius: "12px", border: "1px solid rgba(16, 185, 129, 0.2)" }}>
                          ✓ Safe Read-Only SQL
                        </span>
                      ) : (
                        <span style={{ fontSize: "0.75rem", color: "#ef4444", backgroundColor: "rgba(239, 68, 68, 0.1)", padding: "4px 8px", borderRadius: "12px", border: "1px solid rgba(239, 68, 68, 0.2)", display: "flex", alignItems: "center", gap: "4px" }}>
                          <ShieldAlert size={14} /> Security Restriction
                        </span>
                      )}
                    </div>
                    <pre style={{ backgroundColor: "#0f172a", padding: "12px", borderRadius: "8px", overflowX: "auto", fontSize: "0.85rem", color: "#38bdf8", margin: 0 }}>
                      {currentResponse.sql_query}
                    </pre>
                    {currentResponse.sql_error && (
                      <div style={{ color: "#ef4444", fontSize: "0.85rem", marginTop: "10px" }}>{currentResponse.sql_error}</div>
                    )}
                  </div>
                )}

                {/* Table Output */}
                {currentResponse.query_results && currentResponse.query_results.length > 0 && (
                  <div style={{ backgroundColor: "#1e293b", padding: "20px", borderRadius: "12px", border: "1px solid #334155", display: "flex", flexDirection: "column", gap: "10px" }}>
                    <span style={{ fontWeight: "bold", fontSize: "0.95rem" }}>Data Output</span>
                    <div style={{ overflowX: "auto" }}>
                      <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.85rem", textAlign: "left" }}>
                        <thead>
                          <tr style={{ borderBottom: "1px solid #475569" }}>
                            {Object.keys(currentResponse.query_results[0]).map((key) => (
                              <th key={key} style={{ padding: "8px", color: "#94a3b8" }}>{key}</th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          {currentResponse.query_results.slice(0, 10).map((row, idx) => (
                            <tr key={idx} style={{ borderBottom: "1px solid #334155" }}>
                              {Object.values(row).map((val: any, colIdx) => (
                                <td key={colIdx} style={{ padding: "8px" }}>{String(val)}</td>
                              ))}
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}

                {/* Chart visualization */}
                {currentResponse.visualization && (
                  <div style={{ backgroundColor: "#1e293b", padding: "20px", borderRadius: "12px", border: "1px solid #334155" }}>
                    <span style={{ fontWeight: "bold", fontSize: "0.95rem", display: "block", marginBottom: "15px" }}>Interactive Visualization</span>
                    <div ref={chartRef} style={{ width: "100%", height: "350px" }} />
                  </div>
                )}
              </>
            ) : (
              <div style={{ flex: 1, display: "flex", flexDirection: "column", justifyContent: "center", alignItems: "center", color: "#64748b", gap: "10px" }}>
                <Sparkles size={48} color="#475569" />
                <span>Ask questions or query your database schema using natural language.</span>
              </div>
            )}
          </div>

          {/* Right - AI Insights & Recommendations */}
          <div style={{ flex: 2, borderLeft: "1px solid #1f2937", display: "flex", flexDirection: "column", padding: "20px", gap: "20px", overflowY: "auto", backgroundColor: "#0f172a" }}>
            <div style={{ display: "flex", alignItems: "center", gap: "8px", borderBottom: "1px solid #334155", paddingBottom: "10px" }}>
              <FileText size={20} color="#3b82f6" />
              <h2 style={{ fontSize: "1.1rem", fontWeight: "bold", margin: 0 }}>AI Insights</h2>
            </div>
            {currentResponse?.insights ? (
              <div style={{ fontSize: "0.9rem", color: "#cbd5e1", lineHeight: "1.6", whiteSpace: "pre-line" }}>
                {currentResponse.insights}
              </div>
            ) : (
              <div style={{ color: "#64748b", fontSize: "0.9rem" }}>No insights generated yet. Execute a query to see analysis.</div>
            )}
          </div>
        </div>

        {/* Query Input Box */}
        <div style={{ borderTop: "1px solid #1f2937", padding: "20px 30px", backgroundColor: "#0f172a" }}>
          <form onSubmit={handleQuery} style={{ display: "flex", gap: "12px", position: "relative" }}>
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="e.g. Compare total amount of invoices by status"
              disabled={loading}
              style={{
                flex: 1,
                padding: "14px 20px",
                backgroundColor: "#1e293b",
                color: "#ffffff",
                border: "1px solid #334155",
                borderRadius: "8px",
                fontSize: "0.95rem",
                outline: "none",
              }}
            />
            <button
              type="submit"
              disabled={loading}
              style={{
                padding: "0 24px",
                backgroundColor: "#3b82f6",
                color: "#ffffff",
                border: "none",
                borderRadius: "8px",
                cursor: "pointer",
                fontWeight: "bold",
                display: "flex",
                alignItems: "center",
                gap: "8px",
              }}
            >
              {loading ? <Loader2 size={18} className="animate-spin" /> : <Send size={18} />} Send
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}
