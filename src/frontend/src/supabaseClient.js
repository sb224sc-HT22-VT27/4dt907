/**
 * Supabase client for the frontend.
 *
 * Reads VITE_SUPABASE_URL and VITE_SUPABASE_ANON_KEY from the build-time
 * environment (set in .env or in the Vercel/hosting dashboard).
 *
 * If either variable is missing the exported `supabase` value is null, and
 * callers that check for null will silently skip the DB write so the rest of
 * the app continues to work without a Supabase project configured.
 */
import { createClient } from "@supabase/supabase-js";

const supabaseUrl = import.meta.env.VITE_SUPABASE_URL ?? "";
const supabaseAnonKey = import.meta.env.VITE_SUPABASE_ANON_KEY ?? "";

/**
 * Configured Supabase client, or `null` when env vars are not set.
 *
 * @type {import("@supabase/supabase-js").SupabaseClient | null}
 */
const supabase =
    supabaseUrl && supabaseAnonKey
        ? createClient(supabaseUrl, supabaseAnonKey)
        : null;

export default supabase;
