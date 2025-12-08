import React from 'react';
import { SearchSection } from './components/SearchSection';

export default function App() {
  const baseUrl = "http://scann.ddns.net:8000";

  return (
    <div className="min-h-screen bg-slate-900 text-slate-200 font-sans selection:bg-indigo-500 selection:text-white">
      {/* Header */}
      <header className="border-b border-slate-800 bg-slate-900/50 backdrop-blur-md sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 bg-gradient-to-tr from-indigo-500 to-purple-500 rounded-lg flex items-center justify-center font-bold text-white text-lg">
              S
            </div>
            <span className="font-bold text-xl tracking-tight text-white">ScaNN<span className="text-slate-500 font-normal">Viz</span></span>
          </div>

          <div className="flex items-center gap-4">
             <div className="flex items-center gap-2 px-3 py-1 bg-indigo-900/30 border border-indigo-500/30 rounded-full">
                <span className="w-2 h-2 bg-indigo-500 rounded-full animate-pulse"></span>
                <span className="text-xs text-indigo-400 font-mono">backend trust</span>
             </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="animate-fade-in-up">
           <SearchSection baseUrl={baseUrl} />
        </div>
      </main>
      
      {/* Footer */}
       <footer className="border-t border-slate-800 mt-auto py-6 text-center text-slate-600 text-sm">
          <p>Powered by Google ScaNN & Sentence Transformers</p>
       </footer>
    </div>
  );
}