import React, { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { FileText, Image, Search } from "lucide-react";

export const DemoPreview = () => {
  const [activeTab, setActiveTab] = useState<"description" | "sketch" | "match">("description");
  const [promptText, setPromptText] = useState<string>(
    `"Male suspect, approximately 35-40 years old, with short dark hair and a square\njawline. Notable scar on the left cheek. Brown eyes, approximately 6 feet tall\nwith an athletic build. Last seen wearing a dark jacket."`
  );
  const [generating, setGenerating] = useState<boolean>(false);
  const [resultUrl, setResultUrl] = useState<string | null>(null);
  const [dbEntries, setDbEntries] = useState<any[]>([]);
  const [matchedDbEntry, setMatchedDbEntry] = useState<any | null>(null);
  const [matchedFilename, setMatchedFilename] = useState<string | null>(null);
  const [matchedPromptText, setMatchedPromptText] = useState<string | null>(null);
  const [matchedScore, setMatchedScore] = useState<number | null>(null);
  const [showDetails, setShowDetails] = useState<boolean>(false);

  const downloadProfile = () => {
    if (!matchedDbEntry) return;
    try {
      const dataStr = JSON.stringify(matchedDbEntry, null, 2);
      const blob = new Blob([dataStr], { type: 'application/json' });
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${matchedDbEntry.id || 'profile'}.json`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      window.URL.revokeObjectURL(url);
    } catch (e) {
      console.warn('Download failed', e);
    }
  };

  // Backend base URL (Flask app). Adjust if your server runs on a different host/port.
  const BACKEND_BASE = "http://127.0.0.1:7860";

  const handleGenerate = async () => {
    // Switch to sketch tab and begin generating
    setActiveTab("sketch");
    setGenerating(true);
    setResultUrl(null);

    try {
      // New flow: call backend JSON generator endpoint
      const form = new FormData();
      form.append('prompt', promptText);
      const r = await fetch(`${BACKEND_BASE}/api/generate`, { method: 'POST', body: form });
      const j = await r.json();
      if (j?.error) {
        throw new Error(j.error);
      }

      // j.result_url is relative (e.g. '/static/web_result_...png'), make absolute for UI
      if (j?.result_url) {
        let src = j.result_url as string;
        if (src.startsWith('/')) src = `${BACKEND_BASE}${src}`;
        setResultUrl(src);

        // Add generated result as a synthetic DB entry at the front so it appears in Match grid
        const genEntry = {
          id: `generated-${Date.now()}`,
          _local_image: j.result_url, // store relative so existing UI prepends BACKEND_BASE
          profile_image: j.result_filename || null,
          personal_details: { name: 'Generated result' },
        } as any;
        setDbEntries((prev) => [genEntry, ...(prev || [])]);
      }

      // Use returned match info when present (single request covers both generation and match)
      if (j?.match) {
        console.log('/api/generate match:', j.match);
        const match = j.match;
        setMatchedFilename(match.image_filename || null);
        setMatchedScore(typeof match.score === 'number' ? match.score : (match.score ? Number(match.score) : null));
        setMatchedPromptText(match.matched_txt_content || null);

        const entry = (dbEntries || []).find((e: any) => {
          const p = e.profile_image || '';
          return match.image_filename && p.endsWith(match.image_filename);
        }) || null;
        setMatchedDbEntry(entry);

        if (entry && entry.id) {
          setTimeout(() => {
            const el = document.getElementById(`db-entry-${entry.id}`);
            if (el && typeof el.scrollIntoView === 'function') {
              el.scrollIntoView({ behavior: 'smooth', block: 'center' });
            }
          }, 250);
        }
      } else {
        // no dataset match; clear existing matched Db entry (we still show generated item in grid)
        setMatchedDbEntry(null);
        setMatchedFilename(null);
        setMatchedPromptText(null);
        setMatchedScore(null);
      }
    } catch (e) {
      console.error("Generation failed", e);
      setResultUrl(null);
    } finally {
      setGenerating(false);
    }
  };

  // Load DB entries on mount so match tab can show data
  useEffect(() => {
    (async () => {
      try {
        const r = await fetch(`${BACKEND_BASE}/api/db`);
        const j = await r.json();
        setDbEntries(j || []);
      } catch (e) {
        console.warn('Failed to load DB entries', e);
      }
    })();
  }, []);

  return (
    <section className="py-24 px-6">
      <style>{`
        /* DemoPreview component custom styles */
        .demo-textarea {
          max-height: 420px;
          min-height: 180px;
          padding: 16px;
          font-size: 1rem;
          line-height: 1.4;
          resize: vertical;
          overflow: auto;
        }

        /* WebKit scrollbar */
        .demo-textarea::-webkit-scrollbar {
          width: 12px;
          height: 12px;
        }
        .demo-textarea::-webkit-scrollbar-track {
          background: rgba(15,23,42,0.04);
          border-radius: 9999px;
        }
        .demo-textarea::-webkit-scrollbar-thumb {
          background: rgba(99,102,241,0.6); /* indigo-500 at 60% */
          border-radius: 9999px;
          border: 3px solid rgba(255,255,255,0.0);
        }

        /* Firefox */
        .demo-textarea {
          scrollbar-width: thin;
          scrollbar-color: rgba(99,102,241,0.6) rgba(15,23,42,0.04);
        }

        /* Make generated sketch fit nicely */
        .demo-result-img {
          max-height: 560px;
          width: 100%;
          object-fit: contain;
          border-radius: 8px;
        }

        /* Highlight matched entry with a pulsing ring */
        .matched-pulse {
          position: relative;
        }
        .matched-pulse::after {
          content: "";
          position: absolute;
          left: -6px;
          top: -6px;
          right: -6px;
          bottom: -6px;
          border-radius: 12px;
          box-shadow: 0 0 0 0 rgba(99,102,241,0.35);
          animation: pulseRing 1600ms ease-out 3;
          pointer-events: none;
        }
        @keyframes pulseRing {
          0% {
            box-shadow: 0 0 0 0 rgba(99,102,241,0.45);
          }
          70% {
            box-shadow: 0 0 0 14px rgba(99,102,241,0.0);
          }
          100% {
            box-shadow: 0 0 0 0 rgba(99,102,241,0.0);
          }
        }
        /* Modal styles for matched details */
        .matched-modal-backdrop {
          position: fixed;
          inset: 0;
          background: rgba(2,6,23,0.6);
          display: flex;
          align-items: center;
          justify-content: center;
          z-index: 60;
          padding: 1.25rem;
        }
        .matched-modal {
          width: 100%;
          max-width: 960px;
          background: rgba(15,23,42,0.95); /* dark surface */
          color: #f8fafc; /* near-white text */
          border-radius: 12px;
          box-shadow: 0 12px 40px rgba(2,6,23,0.6);
          overflow: auto;
          border: 1px solid rgba(148,163,184,0.06);
        }
        .matched-details-img { max-height: 420px; width: 100%; object-fit: cover; border-radius: 8px; border: 1px solid rgba(148,163,184,0.06); }
      `}</style>
      <div className="container mx-auto">
        <div className="text-center mb-16">
          <h2 className="text-4xl md:text-5xl font-bold mb-4">
            See the <span className="text-gradient">System in Action</span>
          </h2>
          <p className="text-xl text-muted-foreground max-w-3xl mx-auto">
            Experience how AI transforms witness descriptions into forensic intelligence.
          </p>
        </div>

        <div className="max-w-5xl mx-auto glass-card rounded-2xl overflow-hidden">
          {/* Tabs */}
          <div className="flex border-b border-border/50">
            <button
              onClick={() => setActiveTab("description")}
              className={`flex-1 flex items-center justify-center gap-2 px-6 py-4 transition-all ${
                activeTab === "description"
                  ? "bg-primary/10 border-b-2 border-primary text-primary"
                  : "text-muted-foreground hover:text-foreground hover:bg-secondary/30"
              }`}
            >
              <FileText className="w-5 h-5" />
              <span className="font-medium">Witness Description</span>
            </button>
            <button
              onClick={() => setActiveTab("sketch")}
              className={`flex-1 flex items-center justify-center gap-2 px-6 py-4 transition-all ${
                activeTab === "sketch"
                  ? "bg-primary/10 border-b-2 border-primary text-primary"
                  : "text-muted-foreground hover:text-foreground hover:bg-secondary/30"
              }`}
            >
              <Image className="w-5 h-5" />
              <span className="font-medium">Generated Sketch</span>
            </button>
            <button
              onClick={() => setActiveTab("match")}
              className={`flex-1 flex items-center justify-center gap-2 px-6 py-4 transition-all ${
                activeTab === "match"
                  ? "bg-primary/10 border-b-2 border-primary text-primary"
                  : "text-muted-foreground hover:text-foreground hover:bg-secondary/30"
              }`}
            >
              <Search className="w-5 h-5" />
              <span className="font-medium">Database Match</span>
            </button>
          </div>

          {/* Content */}
          <div className="p-8">
            {activeTab === "description" && (
              <div className="space-y-4 animate-fade-in-up">
                <div className="bg-secondary/50 rounded-lg p-6">
                  <label className="block text-sm font-medium mb-2">Witness Description</label>
                  <textarea
                    value={promptText}
                    onChange={(e) => setPromptText(e.target.value)}
                    className="demo-textarea w-full rounded-md bg-background/50 border"
                    aria-label="Witness description"
                  />
                </div>

                <div className="flex items-center gap-3 text-sm text-muted-foreground">
                  <div className="flex items-center gap-2">
                    <div className="w-2 h-2 rounded-full bg-primary animate-pulse" />
                    <span>Ready to generate sketch from description.</span>
                  </div>
                </div>

                <div className="pt-2">
                  <Button onClick={handleGenerate} className="gap-2">
                    <Image className="w-4 h-4" />
                    Generate Sketch
                  </Button>
                </div>
              </div>
            )}

            {activeTab === "sketch" && (
              <div className="space-y-4 animate-fade-in-up">
                <div className="aspect-video bg-secondary/50 rounded-lg flex items-center justify-center border-2 border-dashed border-primary/30">
                  <div className="text-center space-y-3 w-full">
                    {generating ? (
                      <div className="flex flex-col items-center gap-3">
                        <div className="w-24 h-24 mx-auto rounded-full bg-primary/20 flex items-center justify-center animate-spin">
                          <svg className="w-10 h-10 text-primary" viewBox="0 0 50 50">
                            <circle cx="25" cy="25" r="20" stroke="currentColor" strokeWidth="5" fill="none" strokeLinecap="round" strokeDasharray="31.4 31.4" />
                          </svg>
                        </div>
                        <p className="text-muted-foreground">Generating sketch — please wait...</p>
                      </div>
                    ) : resultUrl ? (
                      <div className="flex flex-col items-center w-full">
                        <img src={resultUrl} alt="Generated sketch" className="demo-result-img rounded-md shadow" />
                        <div className="pt-3">
                          <Button size="sm" variant="outline" className="gap-2" onClick={() => window.open(resultUrl || "", "_blank") }>
                            <Image className="w-4 h-4" />
                            Open Image
                          </Button>
                        </div>
                      </div>
                    ) : (
                      <div className="text-center text-muted-foreground">
                        <div className="w-24 h-24 mx-auto rounded-full bg-primary/20 flex items-center justify-center">
                          <Image className="w-12 h-12 text-primary" />
                        </div>
                        <p className="mt-3">No sketch available. Click Generate on the Description tab.</p>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            )}

            {activeTab === "match" && (
              <div className="space-y-4 animate-fade-in-up">
                {/* If there is a matched DB entry, show it at the top with More Info */}
                {matchedDbEntry ? (
                  <div className="mb-6">
                    <div className={`glass-card p-4 rounded-lg transition-all flex items-center justify-between ${matchedFilename && matchedDbEntry.profile_image && matchedDbEntry.profile_image.endsWith(matchedFilename) ? 'matched-pulse ring-4 ring-primary/40' : ''}`}>
                      <div className="flex items-center gap-4">
                        <div className="w-28 h-28 bg-secondary/50 rounded overflow-hidden">
                          {matchedDbEntry._local_image ? (
                            <img src={`${BACKEND_BASE}${matchedDbEntry._local_image}`} alt={matchedDbEntry.personal_details?.name || 'matched'} className="w-full h-full object-cover" />
                          ) : (
                            <div className="w-full h-full flex items-center justify-center">👤</div>
                          )}
                        </div>
                        <div>
                          <p className="font-medium text-lg">{matchedDbEntry.personal_details?.name}</p>
                          <p className="text-sm text-primary">{matchedDbEntry.personal_details?.age ? `${matchedDbEntry.personal_details.age} yrs` : ''} {matchedDbEntry.personal_details?.state_of_residence ? `• ${matchedDbEntry.personal_details.state_of_residence}` : ''}</p>
                          <p className="text-xs text-muted-foreground">ID: {matchedDbEntry.id}</p>
                          {matchedScore !== null && (
                            <p className="text-xs text-muted-foreground">Match confidence: {(matchedScore * 100).toFixed(1)}%</p>
                          )}
                        </div>
                      </div>
                      <div className="flex items-center gap-3">
                        <Button size="sm" className="gap-2" onClick={() => setShowDetails(!showDetails)}>
                          More information
                        </Button>
                      </div>
                    </div>

                    {showDetails && (
                      <div className="matched-modal-backdrop" role="dialog" aria-modal="true">
                        <div className="matched-modal">
                          <div className="p-4 flex items-center justify-between border-b">
                            <h3 className="text-lg font-semibold">Profile — {matchedDbEntry.personal_details?.name}</h3>
                            <div className="flex items-center gap-2">
                              <Button size="sm" variant="outline" onClick={downloadProfile} className="gap-2">Download profile</Button>
                              <Button size="sm" onClick={() => setShowDetails(false)}>Close</Button>
                            </div>
                          </div>
                          <div className="p-4 max-h-[75vh] overflow-auto">
                            <div className="flex flex-col md:flex-row gap-6">
                              <div className="md:w-1/3 w-full">
                                {matchedDbEntry._local_image ? (
                                  <img src={`${BACKEND_BASE}${matchedDbEntry._local_image}`} alt="full" className="matched-details-img" />
                                ) : null}
                              </div>
                              <div className="md:w-2/3 w-full">
                                <div className="space-y-3 text-sm">
                                  <p><strong>Name:</strong> {matchedDbEntry.personal_details?.name}</p>
                                  <p><strong>ID:</strong> {matchedDbEntry.id}</p>
                                  <p><strong>Age:</strong> {matchedDbEntry.personal_details?.age} — <strong>Gender:</strong> {matchedDbEntry.personal_details?.gender}</p>
                                  <p><strong>DOB:</strong> {matchedDbEntry.personal_details?.date_of_birth}</p>
                                  <p><strong>Nationality:</strong> {matchedDbEntry.personal_details?.nationality}</p>
                                  <p><strong>State:</strong> {matchedDbEntry.personal_details?.state_of_residence}</p>

                                  <h5 className="mt-3 font-medium">Booking Information</h5>
                                  <p><strong>Arrest ID:</strong> {matchedDbEntry.booking_information?.arrest_id}</p>
                                  <p><strong>Location:</strong> {matchedDbEntry.booking_information?.arrest_location}</p>
                                  <p><strong>Time:</strong> {matchedDbEntry.booking_information?.booking_date_time}</p>

                                  <h5 className="mt-3 font-medium">Charges</h5>
                                  {(matchedDbEntry.charges || []).map((c: any, i: number) => (
                                    <div key={i} className="pl-2">
                                      <p className="text-sm"><strong>{c.charge_code}</strong>: {c.description} — <em>{c.status}</em></p>
                                    </div>
                                  ))}

                                  <h5 className="mt-3 font-medium">Administrative Notes</h5>
                                  <p className="text-sm">{matchedDbEntry.administrative_notes?.officer_notes}</p>

                                  {matchedPromptText && (
                                    <>
                                      <h5 className="mt-3 font-medium">Matched Prompt Text</h5>
                                      <pre className="whitespace-pre-wrap text-xs bg-secondary/10 p-2 rounded">{matchedPromptText}</pre>
                                    </>
                                  )}
                                </div>
                              </div>
                            </div>
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                ) : null}

                <div className="grid grid-cols-3 gap-4">
                  {dbEntries && dbEntries.length > 0 ? (
                    dbEntries.slice(0, 9).map((entry, idx) => (
                      <div id={`db-entry-${entry.id || idx}`} key={entry.id || idx} className={`glass-card p-4 rounded-lg hover:glow-effect transition-all ${matchedFilename && entry.profile_image && entry.profile_image.endsWith(matchedFilename) ? 'ring-4 ring-primary/40 matched-pulse' : ''}`}>
                        <div className="aspect-square bg-secondary/50 rounded-lg mb-3 flex items-center justify-center overflow-hidden">
                          {entry._local_image ? (
                            <img src={`${BACKEND_BASE}${entry._local_image}`} alt={entry.personal_details?.name || 'profile'} className="w-full h-full object-cover" />
                          ) : (
                            <span className="text-3xl">👤</span>
                          )}
                        </div>
                        <div className="space-y-1">
                          <p className="font-medium">{entry.personal_details?.name || `Entry ${idx + 1}`}</p>
                          <p className="text-sm text-primary">{entry.personal_details?.age ? `${entry.personal_details.age} yrs` : ''} {entry.personal_details?.state_of_residence ? `• ${entry.personal_details.state_of_residence}` : ''}</p>
                          <p className="text-xs text-muted-foreground">ID: {entry.id}</p>
                        </div>
                      </div>
                    ))
                  ) : (
                    <div className="col-span-3 text-center text-muted-foreground">No database entries available.</div>
                  )}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </section>
  );
};

const Edit3 = ({ className }: { className?: string }) => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    width="24"
    height="24"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
    className={className}
  >
    <path d="M12 20h9" />
    <path d="M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4L16.5 3.5z" />
  </svg>
);
