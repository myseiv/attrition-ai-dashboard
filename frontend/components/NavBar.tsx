'use client'
import Link from 'next/link'
import { usePathname } from 'next/navigation'

const links = [
  { href: '/', label: 'Predict' },
  { href: '/whatif', label: 'What-If' },
  { href: '/global', label: 'Global Importance' },
]

export default function NavBar() {
  const pathname = usePathname()
  return (
    <nav className="bg-slate-900 text-white px-6 py-3 flex items-center gap-8">
      <span className="font-bold text-white tracking-tight">⚡ AttritionAI</span>
      <div className="flex gap-6">
        {links.map(({ href, label }) => (
          <Link
            key={href}
            href={href}
            className={`text-sm transition-colors ${
              pathname === href
                ? 'text-white border-b-2 border-blue-400 pb-0.5'
                : 'text-slate-400 hover:text-white'
            }`}
          >
            {label}
          </Link>
        ))}
      </div>
    </nav>
  )
}
