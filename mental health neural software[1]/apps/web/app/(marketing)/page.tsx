import Link from 'next/link'

export default function LandingPage() {
  return (
    <div className="min-h-screen flex flex-col items-center justify-center p-8 bg-gradient-to-br from-blue-50 to-indigo-100">
      <div className="max-w-2xl w-full text-center space-y-8">
        <h1 className="text-5xl font-bold text-gray-900 mb-4">
          MindWatch
        </h1>
        <p className="text-xl text-gray-700 mb-8">
          Privacy-aware, real-time student well-being assessment
        </p>
        <div className="bg-white rounded-lg shadow-lg p-8 space-y-6">
          <h2 className="text-2xl font-semibold text-gray-800">
            How It Works
          </h2>
          <ul className="text-left space-y-3 text-gray-600">
            <li className="flex items-start">
              <span className="text-green-500 mr-2">✓</span>
              <span>You control what data is collected (face, voice, text, location)</span>
            </li>
            <li className="flex items-start">
              <span className="text-green-500 mr-2">✓</span>
              <span>All processing happens on your device first (privacy-first)</span>
            </li>
            <li className="flex items-start">
              <span className="text-green-500 mr-2">✓</span>
              <span>Only anonymized features are sent to our server</span>
            </li>
            <li className="flex items-start">
              <span className="text-green-500 mr-2">✓</span>
              <span>Get instant feedback with traffic-light scoring</span>
            </li>
          </ul>
        </div>
        <Link
          href="/app"
          className="inline-block bg-indigo-600 text-white px-8 py-4 rounded-lg text-lg font-semibold hover:bg-indigo-700 transition-colors shadow-lg"
        >
          Start Check-in
        </Link>
        <p className="text-sm text-gray-500 mt-4">
          All data collection is optional and requires your explicit consent
        </p>
      </div>
    </div>
  )
}


