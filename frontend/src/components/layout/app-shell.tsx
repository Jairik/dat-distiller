/**
 * App chrome: sidebar navigation + a content area for the routed pages.
 *
 * Local single-user app: no auth, one column of nav. The sidebar lists
 * Projects for quick switching; the Running Jobs bar (#12) mounts here so
 * background work stays visible from anywhere.
 */

import { NavLink, Outlet, useLocation } from 'react-router-dom'
import { PageIn } from '@/components/motion'
import { RunningJobsBanner } from '@/components/jobs/running-jobs-banner'
import { cn } from '@/lib/utils'
import { useProjects } from '@/lib/projects'

function SideLink({ to, label, end }: { to: string; label: string; end?: boolean }) {
  return (
    <NavLink
      to={to}
      end={end}
      className={({ isActive }) =>
        cn(
          'block rounded-md px-3 py-2 text-sm text-muted-foreground transition-colors hover:bg-muted hover:text-foreground',
          isActive && 'bg-muted font-medium text-foreground',
        )
      }
    >
      {label}
    </NavLink>
  )
}

export function AppShell() {
  const projects = useProjects()
  const location = useLocation()
  return (
    <div className="flex min-h-svh">
      <aside className="flex w-60 shrink-0 flex-col gap-6 border-r border-border bg-card/40 p-4">
        <div className="px-2">
          <NavLink to="/" className="text-lg font-semibold tracking-tight">
            Dat Distiller
          </NavLink>
          <p className="text-xs text-muted-foreground">generate · label · train</p>
        </div>
        <nav aria-label="Primary" className="flex flex-col gap-1">
          <SideLink to="/" label="Projects" end />
          <SideLink to="/settings" label="Settings" />
        </nav>
        {(projects.data?.length ?? 0) > 0 && (
          <div className="min-h-0 flex-1 overflow-y-auto">
            <p className="px-3 pb-1 text-xs uppercase tracking-wide text-muted-foreground">
              Projects
            </p>
            <nav aria-label="Projects" className="flex flex-col gap-1">
              {(projects.data ?? []).map((project) => (
                <SideLink
                  key={project.id}
                  to={`/projects/${project.id}/dataset`}
                  label={project.name}
                />
              ))}
            </nav>
          </div>
        )}
      </aside>
      <main className="min-w-0 flex-1">
        <RunningJobsBanner />
        <PageIn key={location.pathname} className="mx-auto w-full max-w-5xl p-6 lg:p-8">
          <Outlet />
        </PageIn>
      </main>
    </div>
  )
}
