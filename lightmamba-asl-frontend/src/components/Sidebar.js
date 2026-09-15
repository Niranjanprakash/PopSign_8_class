import React from 'react';
import { NavLink } from 'react-router-dom';
import { useBackendStatus } from '../hooks/useBackendStatus';

// SVG icon components
const IconDashboard = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/>
    <rect x="14" y="14" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/>
  </svg>
);
const IconVideo = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <polygon points="23 7 16 12 23 17 23 7"/><rect x="1" y="5" width="15" height="14" rx="2" ry="2"/>
  </svg>
);
const IconCamera = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"/>
    <circle cx="12" cy="13" r="4"/>
  </svg>
);
const IconSkeleton = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <circle cx="12" cy="5" r="2"/><path d="M12 7v5"/><path d="M9 12H7a2 2 0 0 0-2 2v1"/><path d="M15 12h2a2 2 0 0 1 2 2v1"/>
    <path d="M9 19l-2 2"/><path d="M15 19l2 2"/><path d="M10 17v-5"/><path d="M14 17v-5"/>
  </svg>
);
const IconCpu = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/>
    <path d="M9 1v3M15 1v3M9 20v3M15 20v3M1 9h3M1 15h3M20 9h3M20 15h3"/>
  </svg>
);
const IconBarChart = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/>
    <line x1="6" y1="20" x2="6" y2="14"/><line x1="2" y1="20" x2="22" y2="20"/>
  </svg>
);
const IconInfo = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/>
    <line x1="12" y1="8" x2="12.01" y2="8"/>
  </svg>
);
const IconBrain = () => (
  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M9.5 2A2.5 2.5 0 0 1 12 4.5v15a2.5 2.5 0 0 1-4.96-.46 2.5 2.5 0 0 1-2.96-3.08 3 3 0 0 1-.34-5.58 2.5 2.5 0 0 1 1.32-4.24 2.5 2.5 0 0 1 1.98-3A2.5 2.5 0 0 1 9.5 2Z"/>
    <path d="M14.5 2A2.5 2.5 0 0 0 12 4.5v15a2.5 2.5 0 0 0 4.96-.46 2.5 2.5 0 0 0 2.96-3.08 3 3 0 0 0 .34-5.58 2.5 2.5 0 0 0-1.32-4.24 2.5 2.5 0 0 0-1.98-3A2.5 2.5 0 0 0 14.5 2Z"/>
  </svg>
);

const IconText = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M17 6.1H3"/><path d="M21 12.1H3"/><path d="M15.1 18H3"/>
  </svg>
);

const NAV_ITEMS = [
  { to: '/',           label: 'Dashboard',         Icon: IconDashboard },
  { to: '/recognize',  label: 'Video Recognition', Icon: IconVideo },
  { to: '/live',       label: 'Live Recognition',  Icon: IconCamera },
  { to: '/sentence',   label: 'Sentence Builder',  Icon: IconText },
  { to: '/skeleton',   label: 'Skeleton Tracking', Icon: IconSkeleton },
  { to: '/model',      label: 'Model Architecture',Icon: IconCpu },
  { to: '/results',    label: 'Research Results',  Icon: IconBarChart },
  { to: '/about',      label: 'About Project',     Icon: IconInfo },
];

export default function Sidebar({ open, onClose }) {
  const { online } = useBackendStatus();

  return (
    <>
      {open && <div className="sidebar-overlay visible" onClick={onClose} />}
      <aside className={`sidebar${open ? ' open' : ''}`}>
        {/* Logo */}
        <div className="sidebar-logo">
          <div className="sidebar-logo-mark">
            <IconBrain />
            <div>
              <div className="sidebar-logo-title">LightMamba-ASL</div>
            </div>
          </div>
          <div className="sidebar-logo-sub">ASL Recognition System</div>
        </div>

        {/* Navigation */}
        <nav className="sidebar-nav">
          <div className="nav-section-label">Navigation</div>
          {NAV_ITEMS.map(({ to, label, Icon }) => (
            <NavLink
              key={to}
              to={to}
              end={to === '/'}
              className={({ isActive }) => `nav-item${isActive ? ' active' : ''}`}
              onClick={onClose}
            >
              <Icon />
              {label}
            </NavLink>
          ))}
        </nav>

        {/* Backend Status */}
        <div className="sidebar-footer">
          <div className="sidebar-status">
            <span className={`pulse-dot ${online === true ? 'green' : online === false ? 'red' : 'amber'}`} />
            <div>
              <div className="sidebar-status-text" style={{
                color: online === true ? 'var(--accent-green)' : online === false ? 'var(--accent-red)' : 'var(--accent-amber)'
              }}>
                {online === true ? 'Backend Online' : online === false ? 'Backend Offline' : 'Connecting...'}
              </div>
              <div className="sidebar-status-label">Flask API · Port 5000</div>
            </div>
          </div>
        </div>
      </aside>
    </>
  );
}
