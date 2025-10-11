set -euo pipefail
SITE_BASE="https://game-over-gauge.netlify.app"
mkdir -p public
LASTMOD="$(date -u +%Y-%m-%d)"
cat > public/sitemap.xml <<XML
<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url>
    <loc>${SITE_BASE}/</loc>
    <lastmod>${LASTMOD}</lastmod>
    <changefreq>daily</changefreq>
    <priority>1.0</priority>
  </url>
  <url>
    <loc>${SITE_BASE}/dashboard.html</loc>
    <lastmod>${LASTMOD}</lastmod>
    <changefreq>daily</changefreq>
    <priority>0.9</priority>
  </url>
  <url>
    <loc>${SITE_BASE}/gauge.json</loc>
    <lastmod>${LASTMOD}</lastmod>
    <changefreq>daily</changefreq>
    <priority>0.5</priority>
  </url>
  <url>
    <loc>${SITE_BASE}/netlify.toml</loc>
    <lastmod>${LASTMOD}</lastmod>
    <changefreq>monthly</changefreq>
    <priority>0.1</priority>
  </url>
</urlset>
XML

cat > public/robots.txt <<ROBOTS
User-agent: *
Allow: /
Sitemap: ${SITE_BASE}/sitemap.xml
ROBOTS

echo "Generated public/sitemap.xml and public/robots.txt"
xmllint --noout public/sitemap.xml 2>/dev/null || echo "xmllint not installed; skipping XML lint."

