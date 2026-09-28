import { Navigate, Route, Routes } from 'react-router-dom'
import { GenerateStep } from '@/components/generate/generate-step'
import { AppShell } from '@/components/layout/app-shell'
import { ProjectsPage } from '@/components/projects/projects-page'
import { DatasetPage } from '@/routes/dataset-page'
import { ProjectPage } from '@/routes/project-page'
import { SectionPlaceholder } from '@/routes/placeholder'
import { SettingsPage } from '@/routes/settings-page'

/**
 * Routes: Projects grid at /, one Project at /projects/:projectId with the
 * four workbench sections, Settings at /settings. `dat-distiller serve`
 * falls any unknown path through to index.html, so deep links work.
 * The Router itself lives in main.tsx (tests inject a MemoryRouter).
 */
export default function App() {
  return (
    <Routes>
      <Route element={<AppShell />}>
        <Route index element={<ProjectsPage />} />
        <Route path="projects/:projectId" element={<ProjectPage />}>
          <Route index element={<Navigate to="dataset" replace />} />
          <Route path="dataset" element={<DatasetPage />} />
          <Route path="generate" element={<GenerateStep />} />
          <Route
            path="label"
            element={
              <SectionPlaceholder
                title="Label"
                blurb="Define Jev Questions, label rows, and clear the Review Queue here."
              />
            }
          />
          <Route
            path="train"
            element={
              <SectionPlaceholder
                title="Train"
                blurb="Train models on labeled data and compare Model Cards here."
              />
            }
          />
        </Route>
        <Route path="settings" element={<SettingsPage />} />
        <Route path="*" element={<ProjectsPage />} />
      </Route>
    </Routes>
  )
}
