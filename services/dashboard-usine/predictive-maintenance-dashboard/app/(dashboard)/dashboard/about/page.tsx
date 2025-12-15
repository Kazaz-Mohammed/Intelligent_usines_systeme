"use client"

import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Info, Code, Database, Zap, Shield, Globe } from "lucide-react"

export default function AboutPage() {
  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">About</h1>
        <p className="text-muted-foreground mt-2">Information about the Predictive Maintenance Dashboard</p>
      </div>

      {/* Application Info */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Info className="h-5 w-5" />
            Application Information
          </CardTitle>
          <CardDescription>Details about this dashboard application</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid gap-4 md:grid-cols-2">
            <div>
              <p className="text-sm font-medium text-muted-foreground">Application Name</p>
              <p className="text-lg font-semibold">Predictive Maintenance Dashboard</p>
            </div>
            <div>
              <p className="text-sm font-medium text-muted-foreground">Version</p>
              <p className="text-lg font-semibold">1.0.0</p>
            </div>
            <div>
              <p className="text-sm font-medium text-muted-foreground">Framework</p>
              <p className="text-lg font-semibold">Next.js 16+</p>
            </div>
            <div>
              <p className="text-sm font-medium text-muted-foreground">UI Library</p>
              <p className="text-lg font-semibold">shadcn/ui</p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Features */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Zap className="h-5 w-5" />
            Key Features
          </CardTitle>
          <CardDescription>Capabilities of the dashboard</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid gap-4 md:grid-cols-2">
            <div className="space-y-2">
              <h4 className="font-semibold">Real-time Monitoring</h4>
              <p className="text-sm text-muted-foreground">
                Live updates via WebSocket for anomalies, RUL predictions, and asset status
              </p>
            </div>
            <div className="space-y-2">
              <h4 className="font-semibold">Anomaly Detection</h4>
              <p className="text-sm text-muted-foreground">
                Advanced ML models for detecting anomalies in sensor data
              </p>
            </div>
            <div className="space-y-2">
              <h4 className="font-semibold">RUL Predictions</h4>
              <p className="text-sm text-muted-foreground">
                Remaining Useful Life predictions with confidence intervals
              </p>
            </div>
            <div className="space-y-2">
              <h4 className="font-semibold">Maintenance Management</h4>
              <p className="text-sm text-muted-foreground">
                Work order tracking and intervention planning
              </p>
            </div>
            <div className="space-y-2">
              <h4 className="font-semibold">KPI Analytics</h4>
              <p className="text-sm text-muted-foreground">
                MTBF, MTTR, OEE, Availability, and Reliability metrics
              </p>
            </div>
            <div className="space-y-2">
              <h4 className="font-semibold">Reports & Export</h4>
              <p className="text-sm text-muted-foreground">
                Generate CSV and PDF reports with custom date ranges
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Technology Stack */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Code className="h-5 w-5" />
            Technology Stack
          </CardTitle>
          <CardDescription>Technologies used in this application</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div>
              <h4 className="font-semibold mb-2">Frontend</h4>
              <div className="flex flex-wrap gap-2">
                <Badge variant="outline">Next.js 16</Badge>
                <Badge variant="outline">React 19</Badge>
                <Badge variant="outline">TypeScript</Badge>
                <Badge variant="outline">Tailwind CSS</Badge>
                <Badge variant="outline">shadcn/ui</Badge>
                <Badge variant="outline">Recharts</Badge>
              </div>
            </div>
            <div>
              <h4 className="font-semibold mb-2">Backend Services</h4>
              <div className="flex flex-wrap gap-2">
                <Badge variant="outline">FastAPI</Badge>
                <Badge variant="outline">Python</Badge>
                <Badge variant="outline">PostgreSQL</Badge>
                <Badge variant="outline">TimescaleDB</Badge>
                <Badge variant="outline">Kafka</Badge>
              </div>
            </div>
            <div>
              <h4 className="font-semibold mb-2">Machine Learning</h4>
              <div className="flex flex-wrap gap-2">
                <Badge variant="outline">PyTorch</Badge>
                <Badge variant="outline">XGBoost</Badge>
                <Badge variant="outline">Scikit-learn</Badge>
                <Badge variant="outline">tsfresh</Badge>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* System Architecture */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Database className="h-5 w-5" />
            System Architecture
          </CardTitle>
          <CardDescription>Microservices architecture overview</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            <div className="flex items-start gap-3 p-3 border rounded-lg">
              <div className="p-2 bg-blue-100 dark:bg-blue-900/20 rounded-md">
                <Globe className="h-4 w-4 text-blue-600 dark:text-blue-400" />
              </div>
              <div>
                <p className="font-semibold">Dashboard Service</p>
                <p className="text-sm text-muted-foreground">
                  Frontend and API gateway for visualization and user interface
                </p>
              </div>
            </div>
            <div className="flex items-start gap-3 p-3 border rounded-lg">
              <div className="p-2 bg-green-100 dark:bg-green-900/20 rounded-md">
                <Zap className="h-4 w-4 text-green-600 dark:text-green-400" />
              </div>
              <div>
                <p className="font-semibold">Feature Extraction Service</p>
                <p className="text-sm text-muted-foreground">
                  Extracts temporal, frequency, and wavelet features from sensor data
                </p>
              </div>
            </div>
            <div className="flex items-start gap-3 p-3 border rounded-lg">
              <div className="p-2 bg-orange-100 dark:bg-orange-900/20 rounded-md">
                <Shield className="h-4 w-4 text-orange-600 dark:text-orange-400" />
              </div>
              <div>
                <p className="font-semibold">Anomaly Detection Service</p>
                <p className="text-sm text-muted-foreground">
                  ML models for detecting anomalies (Isolation Forest, One-Class SVM)
                </p>
              </div>
            </div>
            <div className="flex items-start gap-3 p-3 border rounded-lg">
              <div className="p-2 bg-purple-100 dark:bg-purple-900/20 rounded-md">
                <Zap className="h-4 w-4 text-purple-600 dark:text-purple-400" />
              </div>
              <div>
                <p className="font-semibold">RUL Prediction Service</p>
                <p className="text-sm text-muted-foreground">
                  Predicts Remaining Useful Life using LSTM, GRU, TCN, and XGBoost models
                </p>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Contact & Support */}
      <Card>
        <CardHeader>
          <CardTitle>Support</CardTitle>
          <CardDescription>Get help and support</CardDescription>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-muted-foreground">
            For technical support, feature requests, or bug reports, please contact your system administrator
            or refer to the project documentation.
          </p>
        </CardContent>
      </Card>
    </div>
  )
}

