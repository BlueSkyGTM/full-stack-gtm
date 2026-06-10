<?php
/**
 * Template Name: AI Engineering From Scratch
 *
 * HOW TO USE:
 * 1. Copy the entire site-new/ folder into your theme directory:
 *    /wp-content/themes/your-theme/aischool/
 *
 * 2. Add to your theme's functions.php:
 *    require get_template_directory() . '/aischool-progress.php';
 *
 * 3. In WordPress admin, create these pages with slugs matching exactly:
 *    - "course"   → Assign template: AI Engineering From Scratch
 *    - "catalog"  → Assign template: AI Engineering From Scratch
 *    - "library"  → Assign template: AI Engineering From Scratch
 *    - "projects" → Assign template: AI Engineering From Scratch
 *    - "glossary" → Assign template: AI Engineering From Scratch
 *
 * 4. The plugin handles auth via nonce. You must be logged in for
 *    progress to persist. For a personal site you're always logged in.
 */

if ( ! defined( 'ABSPATH' ) ) exit;

$SITE_DIR_URL = get_template_directory_uri() . '/aischool/site-new';
$SITE_DIR     = get_template_directory()     . '/aischool/site-new';

/* Enqueue shared assets + inject nonce */
aischool_enqueue( $SITE_DIR_URL );

/* Enqueue the screen-specific JS */
$slug = get_post_field( 'post_name', get_post() );
$screen_js = [
    'course'   => 'course.js',
    'catalog'  => 'catalog.js',
    'library'  => 'library.js',
    'projects' => 'projects.js',
    'glossary' => 'glossary.js',
];
if ( isset( $screen_js[ $slug ] ) ) {
    wp_enqueue_script(
        'aischool-screen',
        $SITE_DIR_URL . '/js/' . $screen_js[ $slug ],
        [ 'aischool-ui' ],
        '1.0.0',
        true
    );
}

/* For library/glossary/catalog also load library.css */
if ( in_array( $slug, [ 'library', 'catalog', 'glossary' ] ) ) {
    wp_enqueue_style( 'aischool-lib-extra', $SITE_DIR_URL . '/css/library.css', [ 'aischool-components' ], '1.0.0' );
}

?><!DOCTYPE html>
<html <?php language_attributes(); ?>>
<head>
  <meta charset="<?php bloginfo( 'charset' ); ?>">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title><?php wp_title( '—', true, 'right' ); ?><?php bloginfo( 'name' ); ?></title>
  <?php wp_head(); ?>
</head>
<body>

<?php
/* Render the <main> block from the matching static HTML file */
$html_file = $SITE_DIR . '/' . $slug . '.html';
if ( file_exists( $html_file ) ) {
    $html = file_get_contents( $html_file );
    /* Extract everything from <header through </main> */
    if ( preg_match( '/(<header\b.*<\/main>)/si', $html, $m ) ) {
        echo $m[1];
    }
}
?>

<?php wp_footer(); ?>
</body>
</html>
