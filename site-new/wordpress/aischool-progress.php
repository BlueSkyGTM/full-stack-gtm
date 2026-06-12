<?php
/**
 * Plugin Name: AI Engineering From Scratch — Progress
 * Description: Stores lesson progress for the AI Engineering From Scratch portfolio site.
 * Version: 1.0.0
 */

if ( ! defined( 'ABSPATH' ) ) exit;

/* ---------------------------------------------------------------
   REST endpoint: /wp-json/aischool/v1/progress
   GET  → returns current user's progress JSON
   POST → saves progress JSON, returns { ok: true }
   Solo personal site: uses wp_options keyed by user_id.
--------------------------------------------------------------- */

add_action( 'rest_api_init', function () {
    $args = [
        'permission_callback' => function () {
            return current_user_can( 'read' );
        }
    ];

    register_rest_route( 'aischool/v1', '/progress', array_merge( $args, [
        'methods'  => 'GET',
        'callback' => 'aischool_get_progress',
    ] ) );

    register_rest_route( 'aischool/v1', '/progress', array_merge( $args, [
        'methods'  => 'POST',
        'callback' => 'aischool_save_progress',
    ] ) );
} );

function aischool_option_key() {
    return 'aischool_progress_' . get_current_user_id();
}

function aischool_get_progress() {
    $raw  = get_option( aischool_option_key(), null );
    $data = $raw ? json_decode( $raw, true ) : [ 'v' => 1, 'done' => (object)[], 'days' => [] ];
    return rest_ensure_response( $data );
}

function aischool_save_progress( WP_REST_Request $req ) {
    $body = $req->get_json_params();
    if ( ! is_array( $body ) ) {
        return new WP_Error( 'bad_data', 'Expected JSON object', [ 'status' => 400 ] );
    }
    /* Sanitize: only keep expected keys */
    $clean = [
        'v'         => 1,
        'done'      => isset( $body['done'] ) && is_array( $body['done'] ) ? $body['done'] : [],
        'days'      => isset( $body['days'] ) && is_array( $body['days'] ) ? array_map( 'sanitize_text_field', $body['days'] ) : [],
        'updatedAt' => isset( $body['updatedAt'] ) ? intval( $body['updatedAt'] ) : time() * 1000,
    ];
    update_option( aischool_option_key(), wp_json_encode( $clean ), false );
    return rest_ensure_response( [ 'ok' => true ] );
}

/* ---------------------------------------------------------------
   Enqueue the site JS/CSS and inject the REST nonce.
   Call aischool_enqueue() from your page template or functions.php
--------------------------------------------------------------- */

function aischool_enqueue( $site_dir_url ) {
    $v = '1.0.0';

    wp_enqueue_style(  'aischool-tokens',     $site_dir_url . '/css/tokens.css',     [], $v );
    wp_enqueue_style(  'aischool-components', $site_dir_url . '/css/components.css', [ 'aischool-tokens' ], $v );
    wp_enqueue_style(  'aischool-library',    $site_dir_url . '/css/library.css',    [ 'aischool-components' ], $v );
    wp_enqueue_style(  'aischool-cmdpalette', $site_dir_url . '/css/cmdpalette.css', [ 'aischool-components' ], $v );

    wp_enqueue_script( 'aischool-data',       $site_dir_url . '/js/data.js',         [], $v, true );
    wp_enqueue_script( 'aischool-store',      $site_dir_url . '/js/store.js',        [ 'aischool-data' ], $v, true );
    wp_enqueue_script( 'aischool-game',       $site_dir_url . '/js/game.js',         [ 'aischool-store' ], $v, true );
    wp_enqueue_script( 'aischool-ui',         $site_dir_url . '/js/ui.js',           [ 'aischool-game' ], $v, true );
    wp_enqueue_script( 'aischool-cmdpalette', $site_dir_url . '/js/cmdpalette.js',   [ 'aischool-ui' ], $v, true );

    /* Inject the nonce — store.js checks window.WP_REST_NONCE */
    wp_localize_script( 'aischool-store', 'WP_REST_NONCE', wp_create_nonce( 'wp_rest' ) );
}
