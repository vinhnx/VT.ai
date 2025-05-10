# OAuth Authentication Testing Guide for VT.ai

This document provides step-by-step instructions for testing the OAuth authentication flow implemented in VT.ai Phase 1.

## Prerequisites

1. Make sure you have set up your Supabase project with the necessary tables:

   ```bash
   # Run the setup script
   python scripts/setup_supabase.py
   ```

2. Ensure your `.env` file contains the necessary Supabase credentials:

   ```
   SUPABASE_URL=https://your-project-id.supabase.co
   SUPABASE_KEY=your_supabase_service_key
   ```

3. Configure OAuth providers in your Supabase dashboard:
   - Go to Authentication → Providers
   - Enable and configure GitHub, Google, or other OAuth providers
   - Add proper redirect URLs (usually your application URL)

## Testing the Authentication Flow

### 1. Start the VT.ai Application

```bash
./scripts/run_vtai_app.sh
```

### 2. Access the Login Feature

When the application starts in test mode (without authentication):

- You should see a "Login" button in the UI
- A message indicating you're in "Test Mode" should be displayed

### 3. Test OAuth Authentication

1. Click the "Login" button
2. Follow the instructions to authenticate with GitHub or Google
3. After authentication, you'll receive an access token
4. Copy the token and paste it back in the application when prompted

### 4. Verify Authenticated State

Once logged in:

- You should see a welcome message with your name
- The authentication status should show you as authenticated
- Token usage should be logged to Supabase with your user ID

### 5. Test Logout Function

1. Click the "Logout" button
2. Verify that you're returned to the unauthenticated state
3. Check that the "Login" button reappears

## Troubleshooting

### Authentication Fails

If authentication fails, check:

1. Your Supabase OAuth configuration is correct
2. SUPABASE_URL and SUPABASE_KEY are properly set in your .env file
3. The OAuth provider (GitHub/Google) is properly configured

### Supabase Connection Issues

If you see errors like "Supabase client not initialized":

1. Check your .env file has the correct Supabase URL and key
2. Ensure your Supabase project is active and running
3. Check for any network issues connecting to Supabase

### Token Usage Logging Issues

If token usage is not being logged correctly:

1. Check the usage_logs table exists in your Supabase project
2. Verify the RLS policies are correctly set up
3. Look for any error messages in the application logs

## Next Steps (Phase 2)

In Phase 2, we will:

1. Improve the OAuth flow with a more seamless UI integration
2. Implement subscription management with Stripe
3. Add usage limits based on subscription tiers
4. Create a user profile page

---

*This document is for internal development and testing purposes only.*
