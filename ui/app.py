import gradio as gr
import os
import time
import shutil
import asyncio
import sys
import joblib
import pandas as pd
import numpy as np
import datetime
import json
import glob
import re
import requests
from requests.exceptions import RequestException
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Ensure workspace folders are in Python path
WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(WORKSPACE_ROOT)

from agents.orchestrator import root_agent, run_pipeline
from mcp_server.tools.kaggle_tools import search_kaggle_datasets, download_kaggle_dataset
from google.adk.apps import App

adk_app = App(root_agent=root_agent, name="app")

# Ensure uploads and outputs exist
UPLOADS_DIR = os.path.join(WORKSPACE_ROOT, "uploads")
OUTPUTS_DIR = os.path.join(WORKSPACE_ROOT, "outputs")
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# Custom css for dark theme matching and animations
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono&display=swap');
@import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css');

/* Global Page Styles */
body, html, .gradio-container {
    background-color: #0A0A0F !important;
    color: #F8FAFC !important;
}
body, html, .gradio-container, .gradio-container * {
    font-family: 'Inter', sans-serif !important;
}

/* Reset Gradio Default Component Styles */
.gradio-container {
    padding: 0 !important;
    margin: 0 !important;
    max-width: 100% !important;
}

/* Custom CSS variables to override Gradio orange accents globally */
:root {
    --primary-50: #EEF2F6 !important;
    --primary-100: #E0E7FF !important;
    --primary-200: #C7D2FE !important;
    --primary-300: #A5B4FC !important;
    --primary-400: #818CF8 !important;
    --primary-500: #6366F1 !important; /* Indigo accent primary */
    --primary-600: #4F46E5 !important;
    --primary-700: #4338CA !important;
    --primary-800: #3730A3 !important;
    --primary-900: #312E81 !important;
    --primary-950: #1E1B4B !important;
}

/* Borders and lines */
.gradio-container *, .gradio-container {
    border-color: #2A2A3A !important;
}

/* Disable orange outline / ring on inputs */
input:focus, textarea:focus, select:focus {
    border-color: #6366F1 !important;
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.2) !important;
}

/* Scrollbars */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}
::-webkit-scrollbar-track {
    background: #0A0A0F;
}
::-webkit-scrollbar-thumb {
    background: #2A2A3A;
    border-radius: 9999px;
}
::-webkit-scrollbar-thumb:hover {
    background: #6366F1;
}

#main-content-wrapper {
    max-width: 1200px !important;
    margin: 0 auto !important;
    padding: 80px 40px 80px 40px !important;
}

/* Fixed Top Nav Bar Styles */
#navbar-container {
    position: fixed !important;
    top: 0 !important;
    left: 0 !important;
    width: 100% !important;
    height: 60px !important;
    background: rgba(10, 10, 15, 0.85) !important;
    border-bottom: 1px solid rgba(99, 102, 241, 0.12) !important;
    backdrop-filter: blur(20px) saturate(180%) !important;
    -webkit-backdrop-filter: blur(20px) saturate(180%) !important;
    z-index: 9999 !important;
    padding: 0 40px !important;
    box-sizing: border-box !important;
    display: flex !important;
    align-items: center !important;
    justify-content: space-between !important;
    margin: 0 !important;
}
#navbar-container > div {
    display: flex !important;
    justify-content: space-between !important;
    align-items: center !important;
    width: 100% !important;
}
.nav-left {
    display: flex;
    align-items: center;
    gap: 12px;
}
.nav-logo {
    font-size: 1.3rem;
    font-weight: 800;
    background: linear-gradient(135deg, #F8FAFC 0%, #A5B4FC 50%, #6366F1 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.5px;
}
.nav-badge {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.15), rgba(139, 92, 246, 0.15));
    color: #A5B4FC;
    border: 1px solid rgba(99, 102, 241, 0.25);
    font-size: 0.7rem;
    font-weight: 700;
    padding: 2px 10px;
    border-radius: 9999px;
    letter-spacing: 0.5px;
}
/* Status indicator pills — premium glassmorphism badges */
.status-indicators {
    display: flex;
    gap: 10px;
    align-items: center;
}
.status-indicator {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    font-size: 0.78rem;
    font-weight: 600;
    color: #E2E8F0;
    letter-spacing: 0.3px;
    background: rgba(17, 17, 24, 0.7);
    border: 1px solid rgba(255, 255, 255, 0.06);
    padding: 5px 14px 5px 10px;
    border-radius: 9999px;
    backdrop-filter: blur(12px);
    transition: all 0.3s ease;
}
.status-indicator:hover {
    border-color: rgba(255, 255, 255, 0.12);
    background: rgba(17, 17, 24, 0.9);
    transform: translateY(-1px);
}
.status-indicator.status-online {
    border-color: rgba(16, 185, 129, 0.2);
}
.status-indicator.status-online:hover {
    border-color: rgba(16, 185, 129, 0.4);
    box-shadow: 0 0 15px rgba(16, 185, 129, 0.08);
}
.status-indicator.status-active {
    border-color: rgba(99, 102, 241, 0.2);
}
.status-indicator.status-active:hover {
    border-color: rgba(99, 102, 241, 0.4);
    box-shadow: 0 0 15px rgba(99, 102, 241, 0.08);
}
.status-indicator.status-offline {
    border-color: rgba(239, 68, 68, 0.2);
}
.status-indicator.status-offline:hover {
    border-color: rgba(239, 68, 68, 0.4);
    box-shadow: 0 0 15px rgba(239, 68, 68, 0.08);
}
.dot {
    height: 8px;
    width: 8px;
    border-radius: 50%;
    display: inline-block;
    flex-shrink: 0;
}
.dot-green {
    background-color: #10B981;
    box-shadow: 0 0 6px rgba(16, 185, 129, 0.6), 0 0 12px rgba(16, 185, 129, 0.3);
    animation: dotPulseGreen 2s ease-in-out infinite;
}
.dot-blue {
    background-color: #6366F1;
    box-shadow: 0 0 6px rgba(99, 102, 241, 0.6), 0 0 12px rgba(99, 102, 241, 0.3);
    animation: dotPulseBlue 2s ease-in-out infinite;
}
.dot-red {
    background-color: #EF4444;
    box-shadow: 0 0 6px rgba(239, 68, 68, 0.6), 0 0 12px rgba(239, 68, 68, 0.3);
    animation: dotPulseRed 2s ease-in-out infinite;
}
@keyframes dotPulseGreen {
    0%, 100% { box-shadow: 0 0 4px rgba(16, 185, 129, 0.4), 0 0 8px rgba(16, 185, 129, 0.15); }
    50% { box-shadow: 0 0 8px rgba(16, 185, 129, 0.8), 0 0 16px rgba(16, 185, 129, 0.3); }
}
@keyframes dotPulseBlue {
    0%, 100% { box-shadow: 0 0 4px rgba(99, 102, 241, 0.4), 0 0 8px rgba(99, 102, 241, 0.15); }
    50% { box-shadow: 0 0 8px rgba(99, 102, 241, 0.8), 0 0 16px rgba(99, 102, 241, 0.3); }
}
@keyframes dotPulseRed {
    0%, 100% { box-shadow: 0 0 4px rgba(239, 68, 68, 0.4), 0 0 8px rgba(239, 68, 68, 0.15); }
    50% { box-shadow: 0 0 8px rgba(239, 68, 68, 0.8), 0 0 16px rgba(239, 68, 68, 0.3); }
}

/* Section 2: Hero Section — Premium Redesign */
.hero-wrapper {
    position: relative;
    overflow: hidden;
    border-radius: 24px;
    margin-bottom: 10px;
    background: #0A0A0F;
}

/* Animated floating gradient orbs behind the hero */
.hero-orb {
    position: absolute;
    border-radius: 50%;
    filter: blur(80px);
    opacity: 0.35;
    animation: floatOrb 8s ease-in-out infinite;
    pointer-events: none;
}
.hero-orb-1 {
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, #6366F1 0%, transparent 70%);
    top: -100px;
    left: -80px;
    animation-delay: 0s;
}
.hero-orb-2 {
    width: 350px;
    height: 350px;
    background: radial-gradient(circle, #8B5CF6 0%, transparent 70%);
    top: -50px;
    right: -60px;
    animation-delay: -3s;
}
.hero-orb-3 {
    width: 250px;
    height: 250px;
    background: radial-gradient(circle, #06B6D4 0%, transparent 70%);
    bottom: -40px;
    left: 50%;
    transform: translateX(-50%);
    animation-delay: -5s;
}
@keyframes floatOrb {
    0%, 100% { transform: translateY(0px) scale(1); opacity: 0.3; }
    50% { transform: translateY(-30px) scale(1.08); opacity: 0.45; }
}

/* Subtle dot-grid pattern overlay */
.hero-grid-overlay {
    position: absolute;
    inset: 0;
    background-image: radial-gradient(rgba(99, 102, 241, 0.08) 1px, transparent 1px);
    background-size: 24px 24px;
    pointer-events: none;
    z-index: 1;
}

/* Gradient fade at the edges */
.hero-edge-fade {
    position: absolute;
    inset: 0;
    background: radial-gradient(ellipse at center, transparent 40%, #0A0A0F 85%);
    pointer-events: none;
    z-index: 2;
}

.hero-container {
    position: relative;
    z-index: 3;
    text-align: center;
    padding: 60px 20px 50px 20px;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 18px;
}

/* Eyebrow — glowing badge */
.hero-eyebrow {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    font-size: 0.75rem;
    font-weight: 700;
    color: #F8FAFC;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    margin: 0;
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.12), rgba(139, 92, 246, 0.12));
    border: 1px solid rgba(139, 92, 246, 0.35) !important;
    padding: 8px 22px;
    border-radius: 9999px;
    backdrop-filter: blur(12px);
    box-shadow: 0 0 15px rgba(139, 92, 246, 0.15), inset 0 1px 0 rgba(255, 255, 255, 0.1);
    transition: all 0.3s ease;
}
.hero-eyebrow:hover {
    border-color: #8B5CF6 !important;
    box-shadow: 0 0 25px rgba(139, 92, 246, 0.35), inset 0 1px 0 rgba(255, 255, 255, 0.2);
}
.hero-eyebrow-dot {
    width: 8px;
    height: 8px;
    background: #8B5CF6;
    border-radius: 50%;
    box-shadow: 0 0 10px #8B5CF6, 0 0 20px #8B5CF6;
    animation: pulse 1.5s infinite;
}

/* Hero Title — large with shimmer gradient */
.hero-title {
    font-size: 3.5rem;
    font-weight: 800;
    color: #F8FAFC;
    line-height: 1.15;
    margin: 0;
    letter-spacing: -1px;
    text-shadow: 0 4px 30px rgba(0, 0, 0, 0.5);
}
.hero-title-gradient {
    background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 25%, #06B6D4 50%, #10B981 75%, #6366F1 100%);
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: shimmerGradient 6s linear infinite;
}
@keyframes shimmerGradient {
    0% { background-position: 0% center; }
    100% { background-position: -200% center; }
}

/* Subtext */
.hero-subtext {
    font-size: 1.3rem !important;
    color: #E2E8F0 !important; /* Brighter, fully readable text color */
    max-width: 720px !important;
    margin: 16px auto 0 auto !important;
    line-height: 1.8 !important;
    font-weight: 400 !important;
    text-shadow: 0 2px 10px rgba(0, 0, 0, 0.6) !important;
}
.highlight-text {
    font-weight: 800 !important;
    -webkit-background-clip: text !important;
    background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    display: inline !important;
}
.highlight-text.cyan { background-image: linear-gradient(120deg, #38BDF8, #7DD3FC) !important; }
.highlight-text.indigo { background-image: linear-gradient(120deg, #818CF8, #C7D2FE) !important; }
.highlight-text.purple { background-image: linear-gradient(120deg, #C084FC, #F3E8FF) !important; }
.highlight-text.pink { background-image: linear-gradient(120deg, #F472B6, #FCE7F3) !important; }

/* Feature Pills — glassmorphism with glow on hover */
.hero-pills {
    display: flex;
    justify-content: center;
    gap: 16px;
    margin-top: 25px;
    flex-wrap: wrap;
}
.hero-pill {
    background: rgba(17, 17, 24, 0.45);
    border: 1px solid rgba(255, 255, 255, 0.08);
    color: #F8FAFC;
    font-size: 0.9rem;
    font-weight: 600;
    padding: 10px 22px;
    border-radius: 9999px;
    display: inline-flex;
    align-items: center;
    gap: 10px;
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    backdrop-filter: blur(12px);
    cursor: default;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
}
.pill-svg {
    width: 18px !important;
    height: 18px !important;
    display: inline-block !important;
    vertical-align: middle !important;
    stroke-width: 2.2 !important;
    transition: transform 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
}
.hero-pill:hover .pill-svg {
    transform: scale(1.25) rotate(8deg) !important;
}
.pill-svg.text-amber {
    color: #F59E0B !important;
    filter: drop-shadow(0 0 5px rgba(245, 158, 11, 0.5)) !important;
}
.pill-svg.text-indigo {
    color: #6366F1 !important;
    filter: drop-shadow(0 0 5px rgba(99, 102, 241, 0.5)) !important;
}
.pill-svg.text-rose {
    color: #EF4444 !important;
    filter: drop-shadow(0 0 5px rgba(239, 68, 68, 0.5)) !important;
}
.pill-svg.text-emerald {
    color: #10B981 !important;
    filter: drop-shadow(0 0 5px rgba(16, 185, 129, 0.5)) !important;
}
.hero-pill.pipeline {
    border-color: rgba(245, 158, 11, 0.2) !important;
}
.hero-pill.agents {
    border-color: rgba(99, 102, 241, 0.2) !important;
}
.hero-pill.security {
    border-color: rgba(239, 68, 68, 0.2) !important;
}
.hero-pill.charts {
    border-color: rgba(16, 185, 129, 0.2) !important;
}
.hero-pill.pipeline:hover {
    transform: translateY(-4px) scale(1.03);
    border-color: #F59E0B !important;
    background: rgba(245, 158, 11, 0.08);
    box-shadow: 0 6px 20px rgba(245, 158, 11, 0.25);
}
.hero-pill.agents:hover {
    transform: translateY(-4px) scale(1.03);
    border-color: #6366F1 !important;
    background: rgba(99, 102, 241, 0.08);
    box-shadow: 0 6px 20px rgba(99, 102, 241, 0.25);
}
.hero-pill.security:hover {
    transform: translateY(-4px) scale(1.03);
    border-color: #EF4444 !important;
    background: rgba(239, 68, 68, 0.08);
    box-shadow: 0 6px 20px rgba(239, 68, 68, 0.25);
}
.hero-pill.charts:hover {
    transform: translateY(-4px) scale(1.03);
    border-color: #10B981 !important;
    background: rgba(16, 185, 129, 0.08);
    box-shadow: 0 6px 20px rgba(16, 185, 129, 0.25);
}

/* Section 3: Input Card */
#input-card {
    background: radial-gradient(circle at top left, #13131F, #0E0E15) !important;
    border: 1px solid rgba(99, 102, 241, 0.15) !important;
    border-radius: 20px !important;
    box-shadow: 0 0 50px rgba(99, 102, 241, 0.12), 0 0 100px rgba(139, 92, 246, 0.05) !important;
    padding: 40px !important;
    max-width: 760px !important;
    margin: 0 auto !important;
}

/* File Upload Overrides */
.upload-card-wrapper .file-preview {
    background-color: #1A1A24 !important;
    border: 1px solid #2A2A3A !important;
    border-radius: 12px !important;
}
.upload-card-wrapper .upload-container {
    background-color: #1A1A24 !important;
    border: 2px dashed #2A2A3A !important;
    border-radius: 12px !important;
    height: 140px !important;
    transition: border-color 0.2s, background-color 0.2s !important;
}
.upload-card-wrapper .upload-container:hover {
    border-color: #6366F1 !important;
    background-color: rgba(99, 102, 241, 0.04) !important;
}

/* Dividers */
.input-divider {
    display: flex;
    align-items: center;
    text-align: center;
    margin: 25px 0;
}
.input-divider::before, .input-divider::after {
    content: '';
    flex: 1;
    border-bottom: 1px solid #2A2A3A;
}
.divider-text {
    padding: 0 15px;
    color: #475569;
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 1px;
}

/* Goal Input Styling */
.goal-input-wrapper textarea {
    background-color: #161622 !important;
    border: 1px solid rgba(99, 102, 241, 0.25) !important;
    border-radius: 12px !important;
    color: #F8FAFC !important;
    padding: 16px !important;
    font-size: 1rem !important;
    line-height: 1.5 !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.2) !important;
}
.goal-input-wrapper textarea::placeholder {
    color: #64748B !important;
    font-style: normal !important;
    opacity: 0.8 !important;
}
.goal-input-wrapper textarea:hover {
    border-color: rgba(99, 102, 241, 0.5) !important;
    box-shadow: 0 0 12px rgba(99, 102, 241, 0.15), inset 0 2px 4px rgba(0, 0, 0, 0.2) !important;
}
.goal-input-wrapper textarea:focus {
    border-color: #6366F1 !important;
    background-color: #1A1A2B !important;
    box-shadow: 0 0 20px rgba(99, 102, 241, 0.3), 0 0 0 1px #6366F1 !important;
}
.goal-input-wrapper label span {
    font-size: 0.8rem !important;
    font-weight: 700 !important;
    color: #8B5CF6 !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
    background: linear-gradient(135deg, #8B5CF6 0%, #6366F1 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 6px !important;
    display: inline-block !important;
}

/* Launch Button */
.launch-btn {
    background: linear-gradient(135deg, #6366F1, #8B5CF6, #06B6D4) !important;
    background-size: 200% auto !important;
    border: none !important;
    border-radius: 12px !important;
    height: 54px !important;
    color: #F8FAFC !important;
    font-size: 1.15rem !important;
    font-weight: 700 !important;
    cursor: pointer !important;
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: 0 4px 20px rgba(99, 102, 241, 0.3) !important;
}
.launch-btn:hover {
    background-position: right center !important;
    transform: translateY(-3px) !important;
    box-shadow: 0 8px 25px rgba(99, 102, 241, 0.5), 0 0 15px rgba(6, 182, 212, 0.3) !important;
}
.launch-btn:active {
    transform: translateY(-1px) !important;
}

/* Section 4: Status Panel */
#pipeline-status-panel {
    background-color: #111118 !important;
    border: 1px solid #2A2A3A !important;
    border-radius: 16px !important;
    padding: 30px !important;
    margin-top: 40px !important;
    display: flex !important;
    flex-direction: column !important;
    gap: 20px !important;
}
.panel-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 10px;
}
.panel-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: #F8FAFC;
}
.pulsing-dot {
    height: 10px;
    width: 10px;
    background-color: #6366F1;
    border-radius: 50%;
    animation: pulse 1.5s infinite;
}
@keyframes pulse {
    0% { opacity: 0.6; box-shadow: 0 0 0 0 rgba(99,102,241,0.5); }
    50% { opacity: 1; box-shadow: 0 0 0 8px rgba(99,102,241,0); }
    100% { opacity: 0.6; box-shadow: 0 0 0 0 rgba(99,102,241,0); }
}

/* Agent Status Cards Grid */
#agent-cards-row {
    display: grid !important;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)) !important;
    gap: 15px !important;
}
.agent-card {
    background-color: #1A1A24;
    border: 1px solid #2A2A3A;
    border-radius: 12px;
    padding: 20px;
    display: flex;
    flex-direction: column;
    gap: 15px;
    transition: transform 0.2s, border-color 0.2s;
    min-height: 110px;
    box-sizing: border-box;
}
.agent-card:hover {
    transform: translateY(-2px);
    border-color: #6366F1;
}
.agent-card.running {
    background: linear-gradient(110deg, #1A1A24 8%, #2A2A3A 18%, #1A1A24 33%) !important;
    background-size: 200% 100% !important;
    animation: shimmer 1.5s linear infinite !important;
}
@keyframes shimmer {
    to {
        background-position-x: -200%;
    }
}
.agent-card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.agent-icon {
    font-size: 1.5rem;
}
.agent-name {
    font-size: 0.9rem;
    font-weight: 700;
    color: #F8FAFC;
    margin-bottom: 4px;
}
.agent-status-desc {
    font-size: 0.75rem;
    color: #94A3B8;
}
.status-badge {
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    padding: 3px 8px;
    border-radius: 6px;
}
.badge-waiting {
    background-color: rgba(71, 85, 105, 0.2);
    color: #94A3B8;
}
.badge-running {
    background-color: rgba(99, 102, 241, 0.2);
    color: #6366F1;
}
.badge-done {
    background-color: rgba(16, 185, 129, 0.2);
    color: #10B981;
}
.badge-failed {
    background-color: rgba(239, 68, 68, 0.2);
    color: #EF4444;
}

/* Custom Progress Bar styles */
.custom-progress-wrapper {
    display: flex;
    align-items: center;
    gap: 15px;
    margin: 20px 0 0 0;
    width: 100%;
}
.progress-bar-bg {
    flex: 1;
    height: 8px;
    background-color: #1A1A24;
    border-radius: 9999px;
    overflow: hidden;
}
.progress-bar-fill {
    height: 100%;
    background: linear-gradient(90deg, #6366F1, #8B5CF6);
    border-radius: 9999px;
    width: 0%;
    transition: width 0.4s ease;
}
.progress-percentage {
    font-size: 0.85rem;
    font-weight: 700;
    color: #6366F1;
    min-width: 35px;
    text-align: right;
}

/* Log Box styles */
.logs-container-title {
    font-size: 0.7rem;
    font-weight: 600;
    color: #475569;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-bottom: 8px;
}
.logs-viewport {
    background-color: #0A0A0F;
    border: 1px solid #2A2A3A;
    border-radius: 8px;
    padding: 16px;
    max-height: 180px;
    overflow-y: auto;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    line-height: 1.5;
    box-sizing: border-box;
    scroll-behavior: smooth;
}

/* Collapsed summary bar style */
.completed-summary-bar {
    display: flex;
    align-items: center;
    gap: 12px;
    background-color: #111118;
    border: 1px solid #10B981;
    border-radius: 12px;
    padding: 16px 24px;
    margin-top: 30px;
    box-shadow: 0 0 20px rgba(16, 185, 129, 0.05);
}
.summary-check {
    font-size: 1.25rem;
}
.summary-text {
    font-size: 0.95rem;
    color: #F8FAFC;
}
.summary-divider {
    color: #2A2A3A;
}
.summary-stat {
    font-size: 0.9rem;
    font-weight: 600;
    color: #94A3B8;
}

/* Section 5: Results panel */
#results-panel {
    margin-top: 40px !important;
}
#results-panel h2 {
    font-size: 1.8rem;
    font-weight: 700;
    color: #F8FAFC;
    margin-bottom: 25px;
}

/* Metrics Cards Grid */
.metrics-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 20px;
    margin-bottom: 30px;
}
.metric-card {
    background-color: #111118;
    border: 1px solid #2A2A3A;
    border-radius: 12px;
    padding: 24px;
    text-align: left;
    transition: transform 0.2s, border-color 0.2s;
}
.metric-card:hover {
    transform: translateY(-2px);
}
.metric-val {
    font-size: 2.25rem;
    font-weight: 700;
    margin-bottom: 6px;
}
.card-indigo .metric-val { color: #6366F1; }
.card-purple .metric-val { color: #8B5CF6; }
.card-amber .metric-val { color: #F59E0B; }
.card-green .metric-val { color: #10B981; }

.metric-label {
    font-size: 0.85rem;
    font-weight: 600;
    color: #94A3B8;
    margin-bottom: 4px;
}
.metric-sub {
    font-size: 0.75rem;
    color: #475569;
}

/* Tabs style overrides */
.gradio-container .tab-nav {
    border-bottom: 1px solid #2A2A3A !important;
    background: transparent !important;
    display: flex !important;
    gap: 15px !important;
    margin-bottom: 20px !important;
}
.gradio-container .tab-nav button {
    background: transparent !important;
    border: none !important;
    color: #94A3B8 !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    padding: 10px 16px !important;
    border-bottom: 2px solid transparent !important;
    transition: color 0.2s, border-color 0.2s !important;
}
.gradio-container .tab-nav button.selected {
    color: #F8FAFC !important;
    border-bottom-color: #6366F1 !important;
}

/* Insights Markdown styles */
.results-insights-markdown h1, .results-insights-markdown h2 {
    color: #6366F1 !important;
    font-weight: 700 !important;
    margin-top: 30px !important;
    margin-bottom: 15px !important;
    border-bottom: 1px solid #2A2A3A !important;
    padding-bottom: 8px !important;
}
.results-insights-markdown ol li {
    background-color: #1A1A24 !important;
    border-left: 4px solid #6366F1 !important;
    padding: 16px !important;
    border-radius: 0 8px 8px 0 !important;
    margin-bottom: 12px !important;
    list-style-type: decimal !important;
    margin-left: 20px !important;
}

/* Custom outline download cards grid */
.downloads-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 20px;
    margin-top: 15px;
}
.download-card {
    display: flex;
    align-items: center;
    gap: 15px;
    background-color: transparent;
    border: 1px solid #6366F1;
    border-radius: 10px;
    padding: 16px 20px;
    text-decoration: none !important;
    color: #F8FAFC !important;
    transition: background-color 0.2s, transform 0.2s;
    cursor: pointer;
}
.download-card:hover {
    background-color: #6366F1 !important;
    transform: translateY(-2px);
}
.dl-card-icon {
    font-size: 1.5rem;
}
.dl-card-details {
    display: flex;
    flex-direction: column;
}
.dl-card-name {
    font-size: 0.95rem;
    font-weight: 700;
}
.dl-card-size {
    font-size: 0.75rem;
    color: #94A3B8;
}
.download-card:hover .dl-card-size {
    color: rgba(248, 250, 252, 0.8) !important;
}

/* Predictor */
.predictor-json-input textarea {
    background-color: #1A1A24 !important;
    border: 1px solid #2A2A3A !important;
    border-radius: 8px !important;
    color: #a5f3fc !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.85rem !important;
}
.predict-btn {
    background: linear-gradient(135deg, #6366F1, #8B5CF6) !important;
    border: none !important;
    border-radius: 8px !important;
    color: #F8FAFC !important;
    font-weight: 600 !important;
    transition: transform 0.2s, filter 0.2s !important;
}
.predict-btn:hover {
    filter: brightness(1.1) !important;
    transform: translateY(-1px) !important;
}

/* Footer styles */
.footer-container {
    text-align: center;
    padding: 40px 20px;
    color: #475569;
    font-size: 0.85rem;
    line-height: 1.6;
    border-top: 1px solid #2A2A3A;
    margin-top: 60px;
}
"""

def check_backend_status() -> str:
    """Checks the status of the local MCP Fastapi server and environment configuration for the header indicators."""
    mcp_ok = False
    try:
        r = requests.get("http://127.0.0.1:8000/health", timeout=2)
        if r.status_code == 200:
            mcp_ok = True
    except Exception:
        pass
        
    model_loaded = os.path.exists(os.path.join(OUTPUTS_DIR, "model.joblib"))
    
    mcp_dot = "dot-green" if mcp_ok else "dot-red"
    mcp_text = "MCP Server"
    
    agents_dot = "dot-green" # Agents are always ready when UI loads or pipeline is idle
    agents_text = "Agents Ready"
    
    model_dot = "dot-blue" if model_loaded else "dot-red"
    model_text = "Model Loaded" if model_loaded else "Model Offline"
    
    mcp_class = "status-online" if mcp_ok else "status-offline"
    agents_class = "status-online"
    model_class = "status-active" if model_loaded else "status-offline"
    
    return f"""
    <div class="status-indicators">
        <span class="status-indicator {mcp_class}">
            <span class="dot {mcp_dot}"></span>
            <span>{mcp_text}</span>
        </span>
        <span class="status-indicator {agents_class}">
            <span class="dot {agents_dot}"></span>
            <span>{agents_text}</span>
        </span>
        <span class="status-indicator {model_class}">
            <span class="dot {model_dot}"></span>
            <span>{model_text}</span>
        </span>
    </div>
    """

def make_agent_card(agent_name: str, status: str) -> str:
    """Generates styled HTML for the agent pipeline status cards."""
    icon = "🤖"
    status_desc = "Waiting for pipeline..."
    badge_class = "badge-waiting"
    badge_label = "Waiting"
    card_class = "agent-card"
    
    if status == "running":
        card_class = "agent-card running"

    if agent_name == "EDA Agent":
        icon = "📊"
        if status == "waiting":
            status_desc = "Idle"
        elif status == "running":
            status_desc = "Analyzing distributions..."
            badge_class = "badge-running"
            badge_label = "Running"
        elif status == "done":
            status_desc = "Exploratory analysis complete"
            badge_class = "badge-done"
            badge_label = "Done"
        elif status == "failed":
            status_desc = "Analysis failed"
            badge_class = "badge-failed"
            badge_label = "Failed"
            
    elif agent_name == "ML Agent":
        icon = "⚙️"
        if status == "waiting":
            status_desc = "Idle"
        elif status == "running":
            status_desc = "Evaluating estimators..."
            badge_class = "badge-running"
            badge_label = "Running"
        elif status == "done":
            status_desc = "Best model trained & saved"
            badge_class = "badge-done"
            badge_label = "Done"
        elif status == "failed":
            status_desc = "Training failed"
            badge_class = "badge-failed"
            badge_label = "Failed"
            
    elif agent_name == "Security Agent":
        icon = "🔒"
        if status == "waiting":
            status_desc = "Idle"
        elif status == "running":
            status_desc = "Auditing code & columns..."
            badge_class = "badge-running"
            badge_label = "Running"
        elif status == "done":
            status_desc = "Security clearance approved"
            badge_class = "badge-done"
            badge_label = "Done"
        elif status == "failed":
            status_desc = "Vulnerabilities flagged"
            badge_class = "badge-failed"
            badge_label = "Failed"
            
    elif agent_name == "Report Agent":
        icon = "📄"
        if status == "waiting":
            status_desc = "Idle"
        elif status == "running":
            status_desc = "Compiling findings..."
            badge_class = "badge-running"
            badge_label = "Running"
        elif status == "done":
            status_desc = "Executive report generated"
            badge_class = "badge-done"
            badge_label = "Done"
        elif status == "failed":
            status_desc = "Compilation failed"
            badge_class = "badge-failed"
            badge_label = "Failed"
            
    return f"""
    <div class="{card_class}">
        <div class="agent-card-header">
            <span class="agent-icon">{icon}</span>
            <span class="status-badge {badge_class}">{badge_label}</span>
        </div>
        <div class="agent-card-body">
            <div class="agent-name">{agent_name}</div>
            <div class="agent-status-desc">{status_desc}</div>
        </div>
    </div>
    """

def generate_metric_cards(ml_results: dict, security_results: dict, eda_results: dict) -> str:
    """Generates styled HTML stats cards for the Results Dashboard."""
    score = ml_results.get("accuracy_score")
    model_name = ml_results.get("model_name", "N/A")
    score_str = "N/A"
    if score is not None:
        if 0.0 <= score <= 1.0:
            score_str = f"{score * 100:.1f}%"
        else:
            score_str = f"{score:.2f}"
            
    raw_eda = eda_results.get("raw_results", {}) if isinstance(eda_results, dict) else {}
    basic_info = eda_results.get("basic_info", {}) if isinstance(eda_results, dict) else {}
    if not basic_info and isinstance(raw_eda, dict):
        basic_info = raw_eda.get("basic_info", {})
    row_count = basic_info.get("rows", 0) if isinstance(basic_info, dict) else 0
    row_count_str = f"{row_count:,}" if row_count else "N/A"
    
    # Calculate Key Risk Factors count from ml_results
    importances_list = ml_results.get("feature_importance_top5", [])
    if not importances_list:
        importances_list = ml_results.get("top_features", [])
        
    count = 0
    if importances_list:
        for item in importances_list:
            if isinstance(item, dict):
                imp = item.get("importance", 0)
            elif isinstance(item, (tuple, list)) and len(item) == 2:
                imp = item[1]
            else:
                imp = 0
                
            try:
                imp_val = float(imp)
                if imp_val > 0.05 if imp_val <= 1.0 else imp_val > 5.0:
                    count += 1
            except (ValueError, TypeError):
                pass
                
    risk_factors_count = max(3, min(10, count))
    
    sec_score = security_results.get("security_score", 100)
    sec_issues = len(security_results.get("issues_list", []))
    sec_issues_str = f"{sec_issues} warnings flagged" if sec_issues else "No warnings"

    return f"""
    <div class="metrics-grid">
        <div class="metric-card card-indigo">
            <div class="metric-val">{score_str}</div>
            <div class="metric-label">Model Accuracy</div>
            <div class="metric-sub">{model_name}</div>
        </div>
        <div class="metric-card card-purple">
            <div class="metric-val">{row_count_str}</div>
            <div class="metric-label">Records Analyzed</div>
            <div class="metric-sub">Processed successfully</div>
        </div>
        <div class="metric-card card-amber">
            <div class="metric-val">{risk_factors_count}</div>
            <div class="metric-label">Key Risk Factors</div>
            <div class="metric-sub">Found by ML Agent</div>
        </div>
        <div class="metric-card card-green">
            <div class="metric-val">{sec_score}/100</div>
            <div class="metric-label">Security Score</div>
            <div class="metric-sub">{sec_issues_str}</div>
        </div>
    </div>
    """

def load_charts_html() -> str:
    """Finds all Plotly interactive HTML files inside the outputs folder and embeds them inline using src."""
    import urllib.parse
    html_elements = []
    
    # 1. Heatmap
    heatmap_path = os.path.join(OUTPUTS_DIR, "eda_correlation_heatmap.html")
    if os.path.exists(heatmap_path):
        encoded_path = urllib.parse.quote(heatmap_path, safe='/')
        html_elements.append(f"""
        <div style="margin-bottom: 30px;">
            <h3 style="color: #6366F1; margin-bottom: 10px;">📊 Correlation Heatmap</h3>
            <iframe src="/gradio_api/file={encoded_path}" width="100%" height="550px" style="border:1px solid #2A2A3A; border-radius: 12px; background-color: #0A0A0F;" sandbox="allow-scripts"></iframe>
        </div>
        """)
        
    # 2. Feature Importance
    importance_path = os.path.join(OUTPUTS_DIR, "ml_feature_importance.html")
    if os.path.exists(importance_path):
        encoded_path = urllib.parse.quote(importance_path, safe='/')
        html_elements.append(f"""
        <div style="margin-bottom: 30px;">
            <h3 style="color: #6366F1; margin-bottom: 10px;">🔑 Feature Importance</h3>
            <iframe src="/gradio_api/file={encoded_path}" width="100%" height="550px" style="border:1px solid #2A2A3A; border-radius: 12px; background-color: #0A0A0F;" sandbox="allow-scripts"></iframe>
        </div>
        """)
        
    # 3. Distribution plots
    dist_plots = glob.glob(os.path.join(OUTPUTS_DIR, "eda_dist_*.html"))
    if dist_plots:
        html_elements.append("<h3 style='color: #6366F1; margin-top: 40px; margin-bottom: 15px;'>📈 Feature Distributions</h3>")
        for plot in sorted(dist_plots):
            col_name = os.path.basename(plot).replace("eda_dist_", "").replace(".html", "")
            encoded_path = urllib.parse.quote(plot, safe='/')
            html_elements.append(f"""
            <div style="margin-bottom: 25px;">
                <h4 style="color: #8B5CF6; margin-bottom: 8px;">Distribution of {col_name}</h4>
                <iframe src="/gradio_api/file={encoded_path}" width="100%" height="450px" style="border:1px solid #2A2A3A; border-radius: 12px; background-color: #0A0A0F;" sandbox="allow-scripts"></iframe>
            </div>
            """)
            
    if not html_elements:
        return "<p style='color: #94A3B8;'>No interactive visualization charts generated yet.</p>"
        
    return "\n".join(html_elements)

def generate_download_grid(model_path: str, predictions_path: str, report_path: str) -> str:
    """Generates premium HTML grid of custom styled outline download buttons."""
    import urllib.parse
    
    def get_file_size(p):
        if os.path.exists(p):
            sz = os.path.getsize(p)
            if sz > 1024 * 1024:
                return f"{sz / (1024 * 1024):.1f} MB"
            else:
                return f"{sz / 1024:.1f} KB"
        return "N/A"
        
    elements = []
    if os.path.exists(model_path):
        encoded = urllib.parse.quote(model_path, safe='/')
        sz = get_file_size(model_path)
        elements.append(f"""
        <a href="/gradio_api/file={encoded}" download="model.pkl" class="download-card">
            <span class="dl-card-icon">⚙️</span>
            <div class="dl-card-details">
                <div class="dl-card-name">model.pkl</div>
                <div class="dl-card-size">{sz}</div>
            </div>
        </a>
        """)
        
    if os.path.exists(predictions_path):
        encoded = urllib.parse.quote(predictions_path, safe='/')
        sz = get_file_size(predictions_path)
        elements.append(f"""
        <a href="/gradio_api/file={encoded}" download="predictions.csv" class="download-card">
            <span class="dl-card-icon">📊</span>
            <div class="dl-card-details">
                <div class="dl-card-name">predictions.csv</div>
                <div class="dl-card-size">{sz}</div>
            </div>
        </a>
        """)
        
    if os.path.exists(report_path):
        encoded = urllib.parse.quote(report_path, safe='/')
        sz = get_file_size(report_path)
        elements.append(f"""
        <a href="/gradio_api/file={encoded}" download="report.pdf" class="download-card">
            <span class="dl-card-icon">📄</span>
            <div class="dl-card-details">
                <div class="dl-card-name">report.pdf</div>
                <div class="dl-card-size">{sz}</div>
            </div>
        </a>
        """)
        
    if not elements:
        return "<p style='color: #94A3B8;'>No download assets generated yet.</p>"
        
    return f"""
    <div class="downloads-grid">
        {"".join(elements)}
    </div>
    """

def format_time_counter(seconds: int) -> str:
    """Helper to display Time Elapsed status."""
    mins = seconds // 60
    secs = seconds % 60
    return f"""
    <div style="text-align: right; color: #94A3B8; font-size: 0.85rem; font-weight: 600;">
        Time Elapsed: {mins}:{secs:02d}
    </div>
    """

def format_progress_bar(pct_val: float) -> str:
    """Helper to display custom premium progress slider."""
    pct = int(pct_val * 100)
    return f"""
    <div class="custom-progress-wrapper">
        <div class="progress-bar-bg">
            <div class="progress-bar-fill" style="width: {pct}%"></div>
        </div>
        <div class="progress-percentage">{pct}%</div>
    </div>
    """

def format_logs_html(log_lines: list) -> str:
    """Helper to render live terminal logs inside scrollable styled div."""
    formatted_lines = []
    for line in log_lines:
        color = "#94A3B8" # Default gray
        if "✅" in line or "success" in line.lower() or "complete" in line.lower():
            color = "#10B981" # Green
        elif "⚠️" in line or "[WARNING]" in line or "warning" in line.lower() or "caution" in line.lower():
            color = "#F59E0B" # Amber
        elif "❌" in line or "error" in line.lower() or "failed" in line.lower() or "crash" in line.lower():
            color = "#EF4444" # Red
        elif "🚀" in line or "[System]" in line:
            color = "#6366F1" # Indigo
            
        formatted_lines.append(f'<div style="color: {color}; margin-bottom: 4px;">{line}</div>')
        
    return f"""
    <div class="logs-container-title">LIVE LOGS</div>
    <div class="logs-viewport" id="logs-viewport-id">
        {"".join(formatted_lines)}
    </div>
    <script>
        const viewport = document.getElementById("logs-viewport-id");
        if (viewport) {{
            viewport.scrollTop = viewport.scrollHeight;
        }}
    </script>
    """

def format_completed_summary(elapsed: int, warnings_cnt: int) -> str:
    """Helper to render simple completion header block."""
    return f"""
    <div class="completed-summary-bar">
        <span class="summary-check">✅</span>
        <span class="summary-text">Pipeline Completed successfully in <strong>{elapsed}s</strong></span>
        <span class="summary-divider">|</span>
        <span class="summary-stat">4 Agents Done</span>
        <span class="summary-divider">|</span>
        <span class="summary-stat" style="color: { '#F59E0B' if warnings_cnt > 0 else '#94A3B8' };">{warnings_cnt} Warnings</span>
    </div>
    """

def write_minimal_pdf(filename: str, text: str):
    """Generates a basic, structurally valid PDF containing raw text summaries for local downloads."""
    clean_text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    clean_text = re.sub(r'[\r\n\t]', ' ', clean_text)[:200]
    
    pdf_content = (
        "%PDF-1.4\n"
        "1 0 obj <</Type /Catalog /Pages 2 0 R>> endobj\n"
        "2 0 obj <</Type /Pages /Kids [3 0 R] /Count 1>> endobj\n"
        "3 0 obj <</Type /Page /Parent 2 0 R /Resources <</Font <</F1 <</Type /Font /Subtype /Type1 /BaseFont /Helvetica>>>>>> /MediaBox [0 0 595 842] /Contents 4 0 R>> endobj\n"
        "4 0 obj\n"
        "<</Length 150>>\n"
        "stream\n"
        "BT\n"
        "/F1 14 Tf\n"
        "50 750 Td\n"
        "(DataPilot Executive Report Summary) Tj\n"
        "0 -30 Td\n"
        "/F1 10 Tf\n"
        "50 720 Td\n"
        f"({clean_text[:80]}) Tj\n"
        "0 -20 Td\n"
        f"({clean_text[80:160]}) Tj\n"
        "ET\n"
        "endstream\n"
        "endobj\n"
        "xref\n"
        "0 5\n"
        "0000000000 65535 f\n"
        "0000000009 00000 n\n"
        "0000000056 00000 n\n"
        "0000000111 00000 n\n"
        "0000000282 00000 n\n"
        "trailer <</Size 5 /Root 1 0 R>>\n"
        "startxref\n"
        "470\n"
        "%%EOF\n"
    )
    with open(filename, "wb") as f:
        f.write(pdf_content.encode("latin-1"))

def ensure_download_files(result: dict):
    """Ensures report.pdf, predictions.csv, and model.pkl exist in outputs directory."""
    model_joblib_path = os.path.join(OUTPUTS_DIR, "model.joblib")
    model_pkl_path = os.path.join(OUTPUTS_DIR, "model.pkl")
    if os.path.exists(model_joblib_path):
        shutil.copy(model_joblib_path, model_pkl_path)
            
    pdf_path = os.path.join(OUTPUTS_DIR, "report.pdf")
    if not os.path.exists(pdf_path):
        report_md = result.get("report_markdown", "")
        if report_md:
            write_minimal_pdf(pdf_path, report_md)

# Kaggle Search & Download functions
def perform_kaggle_search(query: str) -> str:
    if not query.strip():
        return "Please enter a search term."
    results = search_kaggle_datasets(query)
    html = ["### 🔍 Kaggle Dataset Search Results", ""]
    for idx, d in enumerate(results):
        html.append(f"**{idx+1}. {d['title']}** (`{d['ref']}`) - {d['size']} ({d['download_count']})")
        html.append(f"> {d['description']}")
        html.append("")
    return "\n".join(html)

def perform_kaggle_download(slug: str):
    if not slug.strip():
        return "Error: Please enter a dataset slug first.", None
    result = download_kaggle_dataset(slug)
    if result["status"] == "success":
        csv_path = result["primary_csv"]
        return f"✅ Dataset successfully downloaded!\nSaved as: {os.path.basename(csv_path)}", csv_path
    else:
        return f"❌ Download failed: {result['message']}", None

# Custom Model Predictor
def predict_custom_model(json_str: str) -> str:
    model_path = os.path.join(OUTPUTS_DIR, "model.joblib")
    if not os.path.exists(model_path):
        return """
        <div class="prediction-box">
            ⚠ Predictor Not Active: No trained model found.<br>
            Please run the pipeline on a dataset first.
        </div>
        """
    try:
        import json as _json
        input_data = _json.loads(json_str)
        if isinstance(input_data, dict):
            input_data = [input_data]
        input_df = pd.DataFrame(input_data)
        
        # Load model pipeline
        model_pipeline = joblib.load(model_path)
        
        # Load metadata sidecar for expected columns and label mapping
        metadata_path = os.path.join(OUTPUTS_DIR, "model_metadata.json")
        expected_cols = None
        target_classes_map = None
        target_name = "outcome"
        problem_type = "classification"
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path) as mf:
                    meta = _json.load(mf)
                expected_cols = meta.get("expected_feature_columns")
                target_classes_map = meta.get("target_classes")  # {"0": "No", "1": "Yes"}
                target_name = meta.get("target", "outcome")
                problem_type = meta.get("problem_type", "classification")
            except Exception:
                pass

        # Load target encoder as fallback
        target_encoder_path = os.path.join(OUTPUTS_DIR, "target_encoder.pkl")
        target_encoder = None
        if os.path.exists(target_encoder_path):
            try:
                target_encoder = joblib.load(target_encoder_path)
            except Exception:
                pass

        # Filter input to only model-expected columns (in correct order)
        if expected_cols:
            available = [c for c in expected_cols if c in input_df.columns]
            missing_input_cols = [c for c in expected_cols if c not in input_df.columns]
            if missing_input_cols:
                # Fill missing cols with 0/median — graceful degradation
                for c in missing_input_cols:
                    input_df[c] = 0
            input_df = input_df[expected_cols]

        def decode_class(pred_idx):
            """Decode numeric class index to human-readable label."""
            idx_str = str(int(pred_idx))
            # Try metadata map first
            if target_classes_map and idx_str in target_classes_map:
                return target_classes_map[idx_str]
            # Try target_encoder
            if target_encoder:
                try:
                    return target_encoder.inverse_transform([pred_idx])[0]
                except Exception:
                    pass
            # Try pipeline classes_
            if hasattr(model_pipeline, 'classes_') and int(pred_idx) < len(model_pipeline.classes_):
                return model_pipeline.classes_[int(pred_idx)]
            return pred_idx

        # Perform prediction
        if hasattr(model_pipeline, "predict_proba"):
            proba = model_pipeline.predict_proba(input_df)[0]
            pred_class_idx = int(np.argmax(proba))
            decoded_prediction = decode_class(pred_class_idx)
            
            # Build class labels for probability display
            if target_classes_map:
                classes_list = [target_classes_map.get(str(i), str(i)) for i in range(len(proba))]
            elif hasattr(model_pipeline, 'classes_'):
                classes_list = [decode_class(c) for c in range(len(proba))]
            else:
                classes_list = [str(i) for i in range(len(proba))]
            
            prob_items = []
            for i in range(len(proba)):
                val = proba[i] * 100
                label = classes_list[i] if i < len(classes_list) else str(i)
                prob_items.append(f"<strong>{label}</strong>: {val:.1f}%")
            prob_str = ", ".join(prob_items)

            # Determine outcome color and title
            decoded_str = str(decoded_prediction).lower()
            pred_is_positive = decoded_str in ["yes", "1", "true", "defaulted", "churned", "positive"] or \
                               str(decoded_prediction) == str(target_classes_map.get("1", "")) if target_classes_map else False

            target_lower = target_name.lower()
            if "churn" in target_lower:
                if pred_is_positive:
                    box_class = "prediction-box prediction-high"
                    title = f"🚨 Churn Risk Detected: {decoded_prediction}"
                else:
                    box_class = "prediction-box prediction-low"
                    title = f"✅ Low Churn Risk: {decoded_prediction}"
            elif "heart" in target_lower or "disease" in target_lower:
                if pred_is_positive:
                    box_class = "prediction-box prediction-high"
                    title = f"❤️ Heart Disease Risk: {decoded_prediction}"
                else:
                    box_class = "prediction-box prediction-low"
                    title = f"✅ No Heart Disease Risk: {decoded_prediction}"
            elif "default" in target_lower:
                if pred_is_positive:
                    box_class = "prediction-box prediction-high"
                    title = f"⚠️ Default Risk: {decoded_prediction}"
                else:
                    box_class = "prediction-box prediction-low"
                    title = f"✅ Low Default Risk: {decoded_prediction}"
            elif "surviv" in target_lower:
                if pred_is_positive:
                    box_class = "prediction-box prediction-low"
                    title = f"✨ Predicted to Survive: {decoded_prediction}"
                else:
                    box_class = "prediction-box prediction-high"
                    title = f"💀 Predicted Not to Survive: {decoded_prediction}"
            else:
                if pred_is_positive:
                    box_class = "prediction-box prediction-high"
                else:
                    box_class = "prediction-box prediction-low"
                title = f"🔮 Prediction: {decoded_prediction}"
            
            return f"""
            <div class="{box_class}">
                <div style="font-size: 1.3rem; font-weight: bold; margin-bottom: 10px;">
                    {title}
                </div>
                <div style="font-size: 1rem; font-weight: normal; color: #d1d5db;">
                    <strong>Probability breakdown:</strong> {prob_str}
                </div>
            </div>
            """
        else:
            # Regression or model with only .predict
            pred = model_pipeline.predict(input_df)[0]
            decoded_prediction = decode_class(pred) if problem_type == "classification" else pred

            style = "background: rgba(99, 102, 241, 0.1); border-color: rgba(99, 102, 241, 0.3); color: #6366F1;"
            return f"""
            <div class="prediction-box" style="{style}">
                <div style="font-size: 1.3rem; font-weight: bold;">
                    🔮 Predicted Value: {decoded_prediction}
                </div>
            </div>
            """
    except Exception as e:
        safe_msg = str(e)
        if len(safe_msg) > 200:
            safe_msg = safe_msg[:200] + "..."
        safe_msg = re.sub(r'/[\w/.-]+/', '[path]/', safe_msg)
        return f"""
        <div class="prediction-box">
            ❌ Prediction failed. Invalid input format or schema mismatch.<br>
            <span style="font-size: 0.85rem; color: #94A3B8;">Detail: {safe_msg}</span>
        </div>
        """
# Main pipeline execution async generator connecting to WebSocket
async def run_data_science_pipeline(file_obj, goal, force_continue: bool = False):
    if not file_obj:
        yield (
            format_time_counter(0),
            format_progress_bar(0.0),
            make_agent_card("EDA Agent", "waiting"),
            make_agent_card("ML Agent", "waiting"),
            make_agent_card("Security Agent", "waiting"),
            make_agent_card("Report Agent", "waiting"),
            format_logs_html(["Error: Please upload or import a CSV dataset first."]),
            gr.update(visible=False), # results_container
            gr.update(visible=False), # security_warning_container
            "", # warning_html
            "", # summary_html
            "", # report_md
            "", # charts_html
            "", # downloads_html
            "", # sample_json
            gr.update(visible=False), # pipeline_status_panel
            gr.update(visible=False), # pipeline_completed_summary
            "" # summary_completed_html
        )
        return
        
    # Input validation and sanitization
    from utils.sanitizer import sanitize_goal, sanitize_csv_file
    try:
        sanitize_csv_file(file_obj.name)
        goal = sanitize_goal(goal)
    except ValueError as e:
        err_msg = f"❌ Input Validation Error: {str(e)}"
        yield (
            format_time_counter(0),
            format_progress_bar(0.0),
            make_agent_card("EDA Agent", "waiting"),
            make_agent_card("ML Agent", "waiting"),
            make_agent_card("Security Agent", "waiting"),
            make_agent_card("Report Agent", "waiting"),
            format_logs_html([err_msg]),
            gr.update(visible=False),
            gr.update(visible=False),
            "", "", "", "", None, None, None, ""
        )
        return

    # Clear previous output files
    for f in os.listdir(OUTPUTS_DIR):
        f_path = os.path.join(OUTPUTS_DIR, f)
        if os.path.isfile(f_path) and not f.startswith('.'):
            try:
                os.remove(f_path)
            except Exception:
                pass

    # Save dataset to uploads
    original_filename = os.path.basename(file_obj.name)
    saved_file_path = os.path.join(UPLOADS_DIR, original_filename)
    shutil.copy(file_obj.name, saved_file_path)
    
    # Initialize UI state vars
    eda_status = "waiting"
    ml_status = "waiting"
    sec_status = "waiting"
    rep_status = "waiting"
    
    warnings_cnt = 0
    progress_val = 0.0
    
    logs = [
        "============================================================",
        "🚀 DATAPILOT: Initiating Multi-Agent Pipeline",
        "============================================================",
        f"• Dataset: {saved_file_path}",
        f"• Goal:    {goal}",
        "------------------------------------------------------------"
    ]
    
    start_time = time.time()
    
    def get_yield_tuple(progress_val, results_visible=False, warning_visible=False, warning_html="", summary_html="", report_md="", charts_html="", sample_json=""):
        elapsed = int(time.time() - start_time)
        
        # Format the downloads grid if complete
        downloads_html = ""
        if results_visible:
            model_pkl_path = os.path.join(OUTPUTS_DIR, "model.pkl")
            predictions_csv_path = os.path.join(OUTPUTS_DIR, "predictions.csv")
            report_pdf_path = os.path.join(OUTPUTS_DIR, "report.pdf")
            downloads_html = generate_download_grid(model_pkl_path, predictions_csv_path, report_pdf_path)
            
        summary_completed_html = ""
        if results_visible:
            summary_completed_html = format_completed_summary(elapsed, warnings_cnt)
            
        # Determine panel visibilities
        running_panel_visible = not results_visible and not warning_visible
        completed_panel_visible = results_visible
        
        return (
            format_time_counter(elapsed),
            format_progress_bar(progress_val),
            make_agent_card("EDA Agent", eda_status),
            make_agent_card("ML Agent", ml_status),
            make_agent_card("Security Agent", sec_status),
            make_agent_card("Report Agent", rep_status),
            format_logs_html(logs),
            gr.update(visible=results_visible),
            gr.update(visible=warning_visible),
            warning_html,
            summary_html,
            report_md,
            charts_html,
            downloads_html,
            sample_json,
            gr.update(visible=running_panel_visible),
            gr.update(visible=completed_panel_visible),
            summary_completed_html
        )

    # Spawn background orchestrator execution task
    pipeline_task = asyncio.create_task(run_pipeline(saved_file_path, goal, force_continue=force_continue))
    
    # Connect to local Fastapi server WebSocket Status Channel
    websocket_url = "ws://127.0.0.1:8000/ws/status"
    ws = None
    ws_connected = False
    
    try:
        import websockets as _ws_lib
        ws = await _ws_lib.connect(websocket_url)
        ws_connected = True
        logs.append("[System] Successfully connected to WebSocket status channel.")
    except Exception as e:
        logs.append(f"[System] Warning: WebSocket status connection failed ({e}). Falling back to local logging.")
        
    yield get_yield_tuple(0.0)

    # Monitor pipeline and WebSocket events
    while not pipeline_task.done():
        progress_updated = False
        if ws_connected and ws:
            try:
                msg_text = await asyncio.wait_for(ws.recv(), timeout=0.5)
                data = json.loads(msg_text)
                agent = data.get("agent")
                message = data.get("message")
                status = data.get("status")
                
                # Parse progress & metrics
                progress_percent = data.get("progress")
                if progress_percent is not None:
                    progress_val = progress_percent / 100.0
                    progress_updated = True
                    
                metrics = data.get("metrics", {})
                if metrics:
                    warnings_cnt = metrics.get("warnings_count", 0)
                
                # Map names
                mapped_agent = agent
                if agent == "DataPilot_EDA_Agent" or agent == "EDA Agent":
                    mapped_agent = "EDA Agent"
                elif agent == "DataPilot_ML_Agent" or agent == "ML Agent":
                    mapped_agent = "ML Agent"
                elif agent == "DataPilot_Security_Agent" or agent == "Security Agent":
                    mapped_agent = "Security Agent"
                elif agent == "DataPilot_Report_Agent" or agent == "Report Agent":
                    mapped_agent = "Report Agent"
                
                timestamp = datetime.datetime.now().strftime("%H:%M:%S")
                logs.append(f"[{timestamp}] [{mapped_agent}] {message}")
                
                if mapped_agent == "EDA Agent":
                    if status == "running":
                        eda_status = "running"
                    elif status == "complete" or status == "done":
                        eda_status = "done"
                    elif status == "error" or status == "failed":
                        eda_status = "failed"
                elif mapped_agent == "ML Agent":
                    if status == "running":
                        ml_status = "running"
                    elif status == "complete" or status == "done":
                        ml_status = "done"
                    elif status == "error" or status == "failed":
                        ml_status = "failed"
                elif mapped_agent == "Security Agent":
                    if status == "running":
                        sec_status = "running"
                    elif status == "complete" or status == "done":
                        sec_status = "done"
                    elif status == "error" or status == "failed":
                        sec_status = "failed"
                elif mapped_agent == "Report Agent":
                    if status == "running":
                        rep_status = "running"
                    elif status == "complete" or status == "done":
                        rep_status = "done"
                    elif status == "error" or status == "failed":
                        rep_status = "failed"
            except asyncio.TimeoutError:
                pass
            except Exception as e:
                logs.append(f"[System] WebSocket link broken: {e}")
                ws_connected = False
        else:
            await asyncio.sleep(0.5)
            
        if not ws_connected or not progress_updated:
            stages_done = sum(1 for s in [eda_status, ml_status, sec_status, rep_status] if s == "done")
            current_running = sum(1 for s in [eda_status, ml_status, sec_status, rep_status] if s == "running")
            progress_val = (stages_done + (0.5 if current_running else 0)) / 4.0
            
        yield get_yield_tuple(progress_val)

    if ws:
        try:
            await ws.close()
        except Exception:
            pass

    # Retrieve execution result
    try:
        result = await pipeline_task
    except Exception as e:
        logs.append(f"❌ Pipeline crashed: {e}")
        yield get_yield_tuple(1.0)
        return

    status = result.get("status")
    
    if status == "warning":
        security_score = result.get("security_score", 100)
        issues = result.get("security_issues", [])
        
        issues_html = ["<ul>"]
        for iss in issues:
            severity = iss.get("severity", "WARNING").upper()
            color = "#EF4444" if severity == "CRITICAL" else "#F59E0B"
            issues_html.append(f"<li style='margin-bottom: 8px;'><span style='color: {color}; font-weight: bold;'>[{severity}]</span> {iss.get('message')} - <em>Fix: {iss.get('suggestion')}</em></li>")
        issues_html.append("</ul>")
        
        warning_card_html = f"""
        <div style="background-color: rgba(239, 68, 68, 0.1); border: 1px solid #ef4444; border-radius: 8px; padding: 20px; color: #fca5a5;">
            <p style="font-size: 1.15rem; font-weight: bold; margin-top: 0;">⚠️ Critical security issues discovered! (Score: {security_score}/100)</p>
            {"".join(issues_html)}
            <p style="font-weight: 600; margin-top: 15px; margin-bottom: 0;">Do you want to ignore this audit and proceed with ML model training?</p>
        </div>
        """
        logs.append("⚠️ Pipeline stopped. Critical security warnings require user permission to proceed.")
        yield get_yield_tuple(
            progress_val=0.75,
            results_visible=False,
            warning_visible=True,
            warning_html=warning_card_html
        )
        return

    if status == "error":
        logs.append(f"❌ Pipeline failed: {result.get('message')}")
        yield get_yield_tuple(1.0)
        return

    # Success
    logs.append("✅ Pipeline completed successfully!")
    eda_status = "done"
    ml_status = "done"
    sec_status = "done"
    rep_status = "done"
    
    ensure_download_files(result)
    summary_cards_html = generate_metric_cards(result.get("ml_results", {}), result.get("security_results", {}), result.get("eda_results", {}))
    charts_html = load_charts_html()
    
    # Generate sample JSON for custom predictor — only include model-expected non-ID columns
    sample_json_str = ""
    try:
        import json as _json
        sample_df = pd.read_csv(saved_file_path)
        metadata_path = os.path.join(OUTPUTS_DIR, "model_metadata.json")
        expected_cols = None
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path) as mf:
                    meta = _json.load(mf)
                expected_cols = meta.get("expected_feature_columns")
            except Exception:
                pass

        if expected_cols:
            # Use only model-expected columns that exist in the dataset
            avail = [c for c in expected_cols if c in sample_df.columns]
            features_df = sample_df[avail]
        else:
            # Fallback: drop target and ID columns
            target_col = result.get("eda_results", {}).get("raw_results", {}).get("patterns_insights", {}).get("detected_target")
            if not target_col:
                target_candidates = ["target", "label", "class", "churn", "default", "status", "y", "sold", "purchased", "admitted", "survived"]
                for col in sample_df.columns:
                    if col.lower() in target_candidates:
                        target_col = col
                        break
            id_cols = []
            for col in sample_df.columns:
                if col == target_col:
                    continue
                if sample_df[col].nunique() == len(sample_df) or (col.lower().endswith("id") and sample_df[col].nunique() > len(sample_df) * 0.8):
                    id_cols.append(col)
            cols_to_drop = [c for c in id_cols + [target_col] if c in sample_df.columns]
            features_df = sample_df.drop(columns=cols_to_drop, errors='ignore')
        
        if not features_df.empty:
            first_row = features_df.iloc[0].to_dict()
            for k, v in list(first_row.items()):
                if isinstance(v, (np.integer, np.floating)):
                    first_row[k] = v.item()
                elif pd.isna(v):
                    first_row[k] = None
            sample_json_str = _json.dumps(first_row, indent=2)
    except Exception as e:
        logger.warning(f"Failed to generate sample JSON input: {e}")
        
    yield get_yield_tuple(
        progress_val=1.0,
        results_visible=True,
        warning_visible=False,
        warning_html="",
        summary_html=summary_cards_html,
        report_md=result.get("report_markdown", "No report content generated."),
        charts_html=charts_html,
        sample_json=sample_json_str
    )

async def run_pipeline_forced(file_obj, goal):
    """Wrapper to force proceed the pipeline bypassing warnings."""
    async for event in run_data_science_pipeline(file_obj, goal, force_continue=True):
        yield event

def abort_pipeline():
    """Resets UI indicators when pipeline is aborted."""
    return (
        gr.update(visible=False), # security_warning_container
        gr.update(visible=False), # pipeline_status_panel
        gr.update(visible=False), # pipeline_completed_summary
        gr.update(visible=False), # results_container
        format_progress_bar(0.0), # progress_bar
        format_time_counter(0),   # time_elapsed_display
        make_agent_card("EDA Agent", "waiting"),
        make_agent_card("ML Agent", "waiting"),
        make_agent_card("Security Agent", "waiting"),
        make_agent_card("Report Agent", "waiting"),
        format_logs_html(["[System] Pipeline aborted by user."])
    )

def launch_ui():
    theme = gr.themes.Default(
        primary_hue="indigo",
        secondary_hue="purple",
        neutral_hue="slate",
    ).set(
        body_background_fill="#0A0A0F",
        body_background_fill_dark="#0A0A0F",
        block_background_fill="#111118",
        block_background_fill_dark="#111118",
        block_border_width="1px",
        block_border_color="#2A2A3A",
        button_primary_background_fill="linear-gradient(135deg, #6366F1, #8B5CF6)",
        button_primary_background_fill_dark="linear-gradient(135deg, #6366F1, #8B5CF6)",
        button_primary_text_color="#ffffff",
        button_primary_text_color_dark="#ffffff",
        block_title_text_color="#ffffff",
        block_title_text_color_dark="#ffffff",
    )
    
    with gr.Blocks(css=CSS, theme=theme, title="DataPilot - Autonomous ML Platform") as demo:
        # Section 1: Navigation Bar
        with gr.Row(elem_id="navbar-container"):
            gr.HTML(
                """
                <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
                <div class="nav-left">
                    <span class="nav-logo">⚡ DataPilot</span>
                    <span class="nav-badge">PRO</span>
                </div>
                """
            )
            backend_status_indicators = gr.HTML(value=check_backend_status())
                
        # Main Dashboard Layout Wrapper
        with gr.Column(elem_id="main-content-wrapper"):
            
            # Section 2: Hero Section — Premium
            gr.HTML(
                """
                <div class="hero-wrapper">
                    <!-- Animated floating gradient orbs -->
                    <div class="hero-orb hero-orb-1"></div>
                    <div class="hero-orb hero-orb-2"></div>
                    <div class="hero-orb hero-orb-3"></div>
                    
                    <!-- Dot grid pattern overlay -->
                    <div class="hero-grid-overlay"></div>
                    
                    <!-- Gradient edge fade -->
                    <div class="hero-edge-fade"></div>
                    
                    <!-- Hero Content -->
                    <div class="hero-container">
                        <p class="hero-eyebrow">
                            <span class="hero-eyebrow-dot"></span>
                            Powered by Google ADK + Multi-Agent AI
                        </p>
                        <h1 class="hero-title">Your Data Science Team<br><span class="hero-title-gradient">— Fully Automated.</span></h1>
                        <p class="hero-subtext">Upload any <span class="highlight-text cyan">CSV</span>. Ask any question. Get <span class="highlight-text indigo">expert-level analysis</span>, <span class="highlight-text purple">trained models</span>, and <span class="highlight-text pink">executive reports</span> — in minutes.</p>
                        <div class="hero-pills">
                            <span class="hero-pill pipeline">
                                <svg class="pill-svg text-amber" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round">
                                    <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon>
                                </svg>
                                6 Min Pipeline
                            </span>
                            <span class="hero-pill agents">
                                <svg class="pill-svg text-indigo" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round">
                                    <rect x="3" y="11" width="18" height="10" rx="2"></rect>
                                    <circle cx="12" cy="5" r="2"></circle>
                                    <path d="M12 7v4"></path>
                                    <line x1="8" y1="16" x2="8" y2="16"></line>
                                    <line x1="16" y1="16" x2="16" y2="16"></line>
                                </svg>
                                4 AI Agents
                            </span>
                            <span class="hero-pill security">
                                <svg class="pill-svg text-rose" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round">
                                    <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path>
                                </svg>
                                Auto Security
                            </span>
                            <span class="hero-pill charts">
                                <svg class="pill-svg text-emerald" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round">
                                    <line x1="18" y1="20" x2="18" y2="10"></line>
                                    <line x1="12" y1="20" x2="12" y2="4"></line>
                                    <line x1="6" y1="20" x2="6" y2="14"></line>
                                </svg>
                                Smart Charts
                            </span>
                        </div>
                    </div>
                </div>
                """
            )
            
            # Section 3: Input Card
            with gr.Column(elem_id="input-card"):
                with gr.Tabs(elem_id="input-tabs") as input_tabs:
                    with gr.TabItem("1. Local Dataset", id="local-input-tab"):
                        with gr.Column(elem_classes=["upload-card-wrapper"]):
                            dataset_input = gr.File(
                                label="Upload Dataset (CSV)",
                                file_types=[".csv"],
                                value=os.path.join(WORKSPACE_ROOT, "uploads", "titanic.csv")
                            )
                            
                    with gr.TabItem("1. Search & Import Kaggle", id="kaggle-input-tab"):
                        with gr.Column(elem_classes=["kaggle-card-wrapper"]):
                            kaggle_query = gr.Textbox(label="Search Kaggle Datasets", placeholder="e.g. churn, titanic, housing...")
                            search_btn = gr.Button("Search Kaggle", variant="secondary")
                            search_results_md = gr.Markdown("Enter search term to list datasets.")
                            
                            kaggle_slug = gr.Textbox(label="Dataset Slug to Download", placeholder="e.g. blastchar/telco-customer-churn")
                            download_btn = gr.Button("Download & Import", variant="secondary")
                            download_status = gr.Textbox(label="Import Status", interactive=False)
                
                gr.HTML(
                    """
                    <div class="input-divider">
                        <span class="divider-text">AND</span>
                    </div>
                    """
                )
                
                with gr.Column(elem_classes=["goal-input-wrapper"]):
                    goal_input = gr.Textbox(
                        label="WHAT DO YOU WANT TO KNOW?",
                        placeholder="e.g. Predict which customers will churn based on their purchase behavior...",
                        lines=3
                    )
                
                launch_btn = gr.Button("Launch DataPilot →", variant="primary", elem_classes=["launch-btn"])
            
            # Section 4: Pipeline Status Panel (Hidden by default)
            with gr.Column(visible=False, elem_id="pipeline-status-panel") as pipeline_status_panel:
                gr.HTML(
                    """
                    <div class="panel-header">
                        <span class="panel-title">Pipeline Running...</span>
                        <span class="pulsing-dot"></span>
                    </div>
                    """
                )
                
                # 4 Agent cards in a row
                with gr.Row(elem_id="agent-cards-row"):
                    eda_badge = gr.HTML(value=make_agent_card("EDA Agent", "waiting"))
                    ml_badge = gr.HTML(value=make_agent_card("ML Agent", "waiting"))
                    sec_badge = gr.HTML(value=make_agent_card("Security Agent", "waiting"))
                    rep_badge = gr.HTML(value=make_agent_card("Report Agent", "waiting"))
                
                # Progress slider and counter
                with gr.Row():
                    progress_bar = gr.HTML(value=format_progress_bar(0.0))
                    time_elapsed_display = gr.HTML(value=format_time_counter(0))
                
                # Live log viewport
                terminal_logs = gr.HTML(value=format_logs_html([]))

            # Collapsed summary row (Hidden by default)
            with gr.Row(visible=False, elem_id="pipeline-completed-summary") as pipeline_completed_summary:
                pipeline_summary_html = gr.HTML()
                
            # Security Warnings Panel (Slack-like permission prompt)
            with gr.Column(visible=False, elem_id="security-warning-panel") as security_warning_container:
                gr.Markdown("### ⚠️ Security Clearance Required")
                security_warning_html = gr.HTML()
                with gr.Row():
                    proceed_btn = gr.Button("Proceed Anyway ⚠️", variant="stop")
                    abort_btn = gr.Button("Abort Pipeline", variant="secondary")

            # Section 5: Results Panel (Hidden by default)
            with gr.Column(visible=False, elem_id="results-panel") as results_container:
                gr.HTML("<h2>📊 DataPilot Pipeline Results</h2>")
                
                # 4 metrics cards
                summary_cards_display = gr.HTML()
                
                with gr.Tabs(elem_id="results-tabs") as results_tabs:
                    # Tab 1: Insights
                    with gr.TabItem("📊 Insights", id="insights-tab"):
                        report_md = gr.Markdown(elem_classes=["results-insights-markdown"])
                        
                    # Tab 2: Charts
                    with gr.TabItem("📈 Charts", id="charts-tab"):
                        charts_display_html = gr.HTML()
                        
                    # Tab 3: Downloads
                    with gr.TabItem("📁 Downloads", id="downloads-tab"):
                        downloads_display_html = gr.HTML()
                        
                    # Tab 4: Live Predictor (Premium predictor)
                    with gr.TabItem("🔮 Live Predictor", id="predictor-tab"):
                        gr.HTML("<h3>🔮 Interactive Model Predictor</h3><p>Test the trained ML model interactively on custom data inputs in JSON format:</p>")
                        with gr.Row():
                            with gr.Column(scale=2):
                                predictor_json = gr.Textbox(
                                    label="INPUT DATA (JSON OBJECT)",
                                    placeholder='{\n  "feature_1": value,\n  "feature_2": value\n}',
                                    lines=10,
                                    value="{}",
                                    max_lines=20,
                                    elem_classes=["predictor-json-input"]
                                )
                                predict_btn = gr.Button("Predict Outcome 🔮", variant="primary", elem_classes=["predict-btn"])
                            with gr.Column(scale=1):
                                prediction_output = gr.HTML(
                                    """
                                    <div class="prediction-box">
                                        Upload a dataset, run the pipeline, and edit the JSON inputs to run a live prediction.
                                    </div>
                                    """
                                )

            # Section 6: Footer
            gr.HTML(
                """
                <div class="footer-container">
                    <div style="width: 60px; height: 2px; background: linear-gradient(90deg, #6366F1, #8B5CF6); margin: 0 auto 16px auto; border-radius: 2px;"></div>
                    <span style="color: #475569;">DataPilot</span>
                    <span style="color: #2A2A3A; margin: 0 8px;">·</span>
                    <span style="color: #334155;">Built with Google ADK</span>
                </div>
                """
            )

        # Wire up Kaggle Search
        search_btn.click(
            fn=perform_kaggle_search,
            inputs=[kaggle_query],
            outputs=[search_results_md]
        )
        
        # Wire up Kaggle Download
        download_btn.click(
            fn=perform_kaggle_download,
            inputs=[kaggle_slug],
            outputs=[download_status, dataset_input]
        )
        
        # Wire up Predictor
        predict_btn.click(
            fn=predict_custom_model,
            inputs=[predictor_json],
            outputs=[prediction_output]
        )
        
        # Wire up Launch Pipeline (Standard)
        launch_btn.click(
            fn=run_data_science_pipeline,
            inputs=[dataset_input, goal_input],
            outputs=[
                time_elapsed_display,
                progress_bar,
                eda_badge,
                ml_badge,
                sec_badge,
                rep_badge,
                terminal_logs,
                results_container,
                security_warning_container,
                security_warning_html,
                summary_cards_display,
                report_md,
                charts_display_html,
                downloads_display_html,
                predictor_json,
                pipeline_status_panel,
                pipeline_completed_summary,
                pipeline_summary_html
            ]
        )

        # Wire up Proceed Anyway Button (Forced execution)
        proceed_btn.click(
            fn=run_pipeline_forced,
            inputs=[dataset_input, goal_input],
            outputs=[
                time_elapsed_display,
                progress_bar,
                eda_badge,
                ml_badge,
                sec_badge,
                rep_badge,
                terminal_logs,
                results_container,
                security_warning_container,
                security_warning_html,
                summary_cards_display,
                report_md,
                charts_display_html,
                downloads_display_html,
                predictor_json,
                pipeline_status_panel,
                pipeline_completed_summary,
                pipeline_summary_html
            ]
        )
        
        # Wire up Abort Button
        abort_btn.click(
            fn=abort_pipeline,
            inputs=[],
            outputs=[
                security_warning_container,
                pipeline_status_panel,
                pipeline_completed_summary,
                results_container,
                progress_bar,
                time_elapsed_display,
                eda_badge,
                ml_badge,
                sec_badge,
                rep_badge,
                terminal_logs
            ]
        )

        # Force dark mode on client-side load and check backend status
        demo.load(
            fn=check_backend_status,
            outputs=[backend_status_indicators],
            js="""() => {
                document.documentElement.classList.add('dark');
            }"""
        )
        
    return demo

if __name__ == "__main__":
    demo = launch_ui()
    port = int(os.environ.get("PORT", 8080))
    is_docker = os.path.exists("/.dockerenv") or os.environ.get("DOCKER_RUN")
    default_host = "0.0.0.0" if is_docker else "127.0.0.1"
    demo.queue().launch(server_name=default_host, server_port=port, allowed_paths=[WORKSPACE_ROOT])
