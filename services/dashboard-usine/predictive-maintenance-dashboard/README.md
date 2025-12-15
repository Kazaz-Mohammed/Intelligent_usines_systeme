# Predictive Maintenance Dashboard

A production-ready Next.js 14+ dashboard application for real-time monitoring and analytics of industrial assets, featuring predictive maintenance capabilities, anomaly detection, and RUL (Remaining Useful Life) predictions.

## Features

- **Authentication & Authorization**
  - JWT-based authentication
  - Protected routes and role-based access
  - Session management with automatic logout

- **Real-time Monitoring**
  - WebSocket integration for live updates
  - Real-time anomaly detection alerts
  - Live RUL predictions and confidence intervals
  - Connection status indicator

- **Asset Management**
  - Comprehensive asset inventory with filtering and pagination
  - Detailed asset views with historical data
  - Asset location tracking with coordinates
  - Status tracking (Operational, Warning, Critical, Maintenance, Offline)

- **Anomaly Detection**
  - Anomaly list with severity filtering
  - Anomaly heatmap visualization
  - Real-time anomaly alerts
  - Historical anomaly tracking

- **RUL Predictions**
  - Remaining Useful Life predictions with confidence intervals
  - RUL trend visualization
  - Asset-wise predictions and comparisons
  - Confidence level indicators

- **Maintenance Management**
  - Work order management (Kanban-style)
  - Intervention planning and tracking
  - Priority and status management
  - Assignment tracking
  - Historical maintenance records

- **Key Performance Indicators (KPIs)**
  - MTBF (Mean Time Between Failures)
  - MTTR (Mean Time To Repair)
  - OEE (Overall Equipment Effectiveness)
  - Availability and Reliability metrics
  - Real-time KPI updates

- **Reports & Export**
  - CSV export functionality
  - PDF report generation
  - Custom date range selection
  - Asset-specific filtering
  - Report templates

- **GIS Integration**
  - Asset location mapping
  - Geographic asset search
  - Location-based filtering

- **Responsive Design**
  - Mobile-friendly interface
  - Tablet optimization
  - Desktop full features
  - Dark/Light mode toggle

## Technology Stack

- **Frontend**
  - Next.js 14+ (App Router)
  - React 19+
  - TypeScript (strict mode)
  - Tailwind CSS v4
  - shadcn/ui components
  - Recharts for data visualization

- **State Management**
  - React Context for authentication
  - React Hooks for local state
  - SWR (for future client-side caching)

- **HTTP & Real-time**
  - Axios for REST API calls
  - WebSocket for real-time updates
  - Auto-reconnect logic

- **Utilities**
  - Date-fns for date formatting
  - Zod for validation (when needed)
  - React Hook Form for forms (when needed)

## Getting Started

### Prerequisites

- Node.js 18+ 
- npm or yarn
- Backend API running on `http://localhost:8091` (configurable)

### Installation

1. **Clone the repository**
\`\`\`bash
git clone <repository-url>
cd predictive-maintenance-dashboard
\`\`\`

2. **Install dependencies**
\`\`\`bash
npm install
\`\`\`

3. **Configure environment variables**
\`\`\`bash
cp .env.example .env.local
\`\`\`

4. **Update `.env.local` with your API configuration**
\`\`\`env
# API Configuration
NEXT_PUBLIC_API_BASE_URL=http://localhost:8091/api/v1
NEXT_PUBLIC_WS_URL=ws://localhost:8091/ws/dashboard
\`\`\`

5. **Run the development server**
\`\`\`bash
npm run dev
\`\`\`

6. **Open your browser**
Navigate to `http://localhost:3000`

## Default Login Credentials

For demo purposes, the application accepts any email/password combination:

- **Email**: `demo@example.com` (or any email)
- **Password**: `any` (or any password)

## Project Structure

\`\`\`
app/
├── (auth)/
│   └── login/
│       └── page.tsx
├── (dashboard)/
│   ├── layout.tsx
│   └── dashboard/
│       ├── page.tsx
│       ├── assets/
│       │   ├── page.tsx
│       │   └── [id]/
│       │       └── page.tsx
│       ├── anomalies/
│       │   └── page.tsx
│       ├── rul/
│       │   └── page.tsx
│       ├── interventions/
│       │   └── page.tsx
│       ├── map/
│       │   └── page.tsx
│       └── reports/
│           └── page.tsx
├── layout.tsx
├── page.tsx
└── globals.css

components/
├── layout/
│   ├── sidebar.tsx
│   └── header.tsx
├── charts/
│   ├── rul-trend-chart.tsx
│   ├── anomaly-heatmap.tsx
│   ├── status-pie-chart.tsx
│   └── kpi-gauge.tsx
├── tables/
│   └── assets-table.tsx
└── ui/
    └── (shadcn components)

context/
└── auth-context.tsx

hooks/
├── useWebSocket.ts
├── useAssets.ts
└── (other data hooks)

lib/
├── api.ts
├── auth.ts
├── websocket.ts
└── utils.ts

types/
└── index.ts
\`\`\`

## API Integration

The dashboard connects to a backend API with the following base configuration:

- **Base URL**: `http://localhost:8091/api/v1`
- **WebSocket URL**: `ws://localhost:8091/ws/dashboard`
- **Auth Header**: `Authorization: Bearer {token}`

### API Endpoints

**Assets**
- `GET /assets` - List assets
- `GET /assets/{asset_id}` - Asset details
- `GET /assets/{asset_id}/features` - Asset features

**Anomalies**
- `GET /anomalies` - List anomalies
- `GET /anomalies/{anomaly_id}` - Anomaly details

**RUL Predictions**
- `GET /rul` - List predictions
- `GET /rul/{asset_id}/latest` - Latest RUL

**Interventions**
- `GET /interventions` - List interventions

**KPIs**
- `GET /kpis/summary` - KPI metrics
- `GET /kpis/trend/{metric_name}` - Metric trends

**GIS**
- `GET /gis/locations` - Asset locations
- `GET /gis/nearby` - Nearby assets

**Export**
- `GET /export/csv` - CSV export
- `GET /export/pdf` - PDF export

## WebSocket Messages

The dashboard receives real-time updates via WebSocket:

\`\`\`typescript
interface WebSocketMessage {
  type: 'feature_update' | 'anomaly_detected' | 'rul_prediction' | 'pong' | 'connection'
  asset_id?: string
  data?: any
  timestamp: string
}
\`\`\`

## Authentication

The application uses JWT-based authentication with the following flow:

1. User enters credentials on login page
2. Credentials sent to `/api/auth/login` endpoint
3. Backend returns JWT token and user info
4. Token stored in localStorage
5. Token included in all subsequent API requests
6. Token automatically refreshed or logout triggered on 401 response

## Real-time Updates

WebSocket connection automatically:
- Connects on app initialization
- Reconnects after 3 seconds if disconnected
- Displays connection status in header
- Updates dashboard data on real-time events
- Merges updates with existing data without replacing

## Customization

### Theme Colors

Edit `app/globals.css` to customize the color scheme:

\`\`\`css
:root {
  --primary: oklch(0.4 0.15 261.4);
  --success: oklch(0.65 0.16 130);
  --warning: oklch(0.85 0.18 60);
  --critical: oklch(0.58 0.24 27.3);
  /* ... more colors ... */
}
\`\`\`

### Dark Mode

Toggle dark mode using the sun/moon icon in the header. Preference is stored in browser.

### Responsive Breakpoints

The dashboard uses Tailwind CSS breakpoints:
- `sm`: 640px
- `md`: 768px
- `lg`: 1024px
- `xl`: 1280px
- `2xl`: 1536px

## Development Tips

- Use React DevTools for component inspection
- Check browser console for WebSocket connection logs
- Mock data is used when API is unavailable
- All date formatting uses ISO 8601 standard
- TypeScript strict mode enabled for type safety

## Building for Production

\`\`\`bash
npm run build
npm start
\`\`\`

## Performance Optimization

- Code splitting via Next.js dynamic imports
- Image optimization with Next.js Image component
- CSS-in-JS optimization via Tailwind
- WebSocket for real-time instead of polling
- API response caching via React Query patterns

## Security Considerations

- JWT tokens stored securely
- Auto-logout on token expiry
- Protected routes require authentication
- CORS configured for API requests
- Input validation on all forms

## Troubleshooting

**WebSocket connection fails**
- Check backend is running on correct URL
- Verify `NEXT_PUBLIC_WS_URL` in `.env.local`
- Check browser console for connection errors

**API calls return 401**
- Token may have expired
- Try logging out and back in
- Check localStorage for auth token

**Dashboard doesn't load**
- Ensure you're authenticated (redirected to login otherwise)
- Check API backend is running
- Verify environment variables are set correctly

## License

MIT

## Support

For issues or questions, contact the development team or open an issue in the repository.
