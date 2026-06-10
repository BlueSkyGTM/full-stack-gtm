# WordPress Installation — AI Engineering From Scratch

Static site → WordPress in 5 steps. No build tool. No npm. Copy files, activate plugin, create pages.

---

## What the two files do

| File | Role |
|------|------|
| `aischool-progress.php` | Plugin — REST endpoint at `/wp-json/aischool/v1/progress` for progress GET/POST, keyed per user in `wp_options`. Also provides `aischool_enqueue()` for loading site assets. |
| `page-aischool.php` | Page template — reads the static HTML files, extracts `<header>` + `<main>`, enqueues the right screen JS per page slug. |

Progress auto-syncs to WP when the user is logged in. Falls back to localStorage when logged out (e.g. preview mode).

---

## Step 1 — Copy site files into your theme

```
wp-content/themes/your-theme/
├── aischool/
│   └── site-new/          ← copy the entire site-new/ folder here
│       ├── css/
│       ├── js/
│       ├── index.html
│       ├── course.html
│       ├── catalog.html
│       ├── library.html
│       ├── projects.html
│       ├── glossary.html
│       ├── roadmap.html
│       └── lesson.html
├── aischool-progress.php  ← copy from wordpress/
├── page-aischool.php      ← copy from wordpress/
└── functions.php          (yours — add one line, see Step 2)
```

Do not copy `screenshots/`, `uploads/`, or `AI School Wireframes.html` — they are not needed.

---

## Step 2 — Load the plugin from functions.php

Add one line anywhere in your theme's `functions.php`:

```php
require get_template_directory() . '/aischool-progress.php';
```

This registers the REST endpoint and makes `aischool_enqueue()` available to the page template.

---

## Step 3 — Activate Google Fonts (optional)

The site uses two Google Fonts loaded in the static HTML `<head>`. WordPress strips the static `<head>`, so add them to `functions.php` if you want the pixel font:

```php
add_action( 'wp_enqueue_scripts', function () {
    wp_enqueue_style(
        'aischool-fonts',
        'https://fonts.googleapis.com/css2?family=VT323&family=JetBrains+Mono:wght@400;500;700&display=swap',
        [],
        null
    );
} );
```

---

## Step 4 — Create pages in WordPress admin

Go to **Pages → Add New** for each of these. Slug must match exactly.

| Page title | Slug | Template |
|------------|------|----------|
| Course | `course` | AI Engineering From Scratch |
| Catalog | `catalog` | AI Engineering From Scratch |
| Library | `library` | AI Engineering From Scratch |
| Projects | `projects` | AI Engineering From Scratch |
| Glossary | `glossary` | AI Engineering From Scratch |

To set the template: in the page editor sidebar → **Page Attributes → Template → AI Engineering From Scratch**.

Leave the page content blank — the template renders everything from the static HTML files.

**Home and Roadmap:** Serve `index.html` and `roadmap.html` as static files or handle separately. They have no screen-specific JS that needs the nonce, so a simple iframe or static embed works.

---

## Step 5 — Verify progress persistence

1. Log in to WordPress.
2. Open `/course` and tick a lesson complete.
3. In browser DevTools → Network, confirm a POST to `/wp-json/aischool/v1/progress` returns `{"ok":true}`.
4. Hard-refresh — the lesson should still show as complete (loaded from WP, not localStorage).

If you see a 401, the nonce is missing. Check that `aischool-progress.php` is required in `functions.php` and the page is using the correct template.

---

## How progress works

```
User action → store.toggle() → localAdapter.write() (instant, sync)
                             → restAdapter.write()  (background POST to WP)

Page load   → store.init()  → window.WP_REST_NONCE present?
                             → yes: fetch WP, merge with localStorage
                             → no:  use localStorage only
```

`window.WP_REST_NONCE` is injected by `wp_localize_script()` in `aischool_enqueue()`. The page template calls `aischool_enqueue()` automatically. No code changes needed.

---

## Lesson reader (`lesson.html`)

The lesson reader fetches lesson markdown from the upstream GitHub repo at runtime (`https://github.com/rohitg00/ai-engineering-from-scratch/...`). This is a client-side fetch — WordPress hosting does not affect it. If the upstream repo moves, update the URL derivation in `js/lesson.js` → `loadLesson()`.

To serve lesson.html as a WP page, add `'lesson' => 'lesson.js'` to the `$screen_js` array in `page-aischool.php` and create a page with slug `lesson`.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| Blank page | Template not assigned | Set template in Page Attributes |
| Progress not saving | `aischool-progress.php` not loaded | Add `require` to functions.php |
| 401 on POST | Nonce missing or expired | Confirm `aischool_enqueue()` is called; nonce expires after 24h |
| CSS missing | Wrong path in `$SITE_DIR_URL` | Verify site-new is at `themes/your-theme/aischool/site-new/` |
| Pixel font missing | Google Fonts not enqueued | Add Step 3 to functions.php |
