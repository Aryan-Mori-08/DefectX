# DefectX Authentication System

## Overview
Complete authentication and authorization system with role-based access control.

## User Roles

### 1. Admin
- **Full system access**
- User management (add/delete users)
- Access to all features
- Product onboarding and configuration

**Default Login:**
- Username: `admin`
- Password: `admin123`

### 2. Operator
- **Quality control access**
- Inspection operations
- Product onboarding
- Results viewing
- Feedback submission

**Default Login:**
- Username: `operator1`
- Password: `operator123`

### 3. Employee
- **Basic inspection access**
- Inspection operations only
- View results (read-only)

**Default Login:**
- Username: `employee1`
- Password: `employee123`

## Features

### 🔐 Login Page
- Clean, branded login interface
- Secure password hashing (SHA-256)
- Default credentials display
- Error handling

### 🎨 Navigation Bar
- User name and role display
- Color-coded role badges
- DefectX branding
- Clean, professional design

### 👥 User Management (Admin Only)
- Add new users
- Delete users (except admin)
- View all users
- Assign roles

### 🔒 Role-Based Access Control
- **Admin**: All tabs + User Management
- **Operator**: Inspection, Onboarding, Results, Feedback
- **Employee**: Inspection, Results View only

## Security Features

- ✅ Password hashing with SHA-256
- ✅ Session-based authentication
- ✅ Role-based permissions
- ✅ Secure user storage (JSON)
- ✅ Admin-only user management
- ✅ Cannot delete admin account

## File Structure

```
streamlit_ui/
├── utils/
│   └── auth.py          # Authentication logic
├── config/
│   └── users.json       # User database (auto-generated)
└── app.py              # Main app with auth integration
```

## How It Works

1. **First Launch**: Default users are created automatically
2. **Login**: Users authenticate with username/password
3. **Session**: Authenticated session stored in Streamlit state
4. **Authorization**: Each page checks user permissions
5. **Logout**: Clear session and return to login

## Customization

### Add New Roles
Edit `role_hierarchy` in `auth.py`:
```python
role_hierarchy = {
    "admin": 3,
    "operator": 2,
    "employee": 1,
    "custom_role": 2  # Add your role here
}
```

### Change Default Passwords
For security, change default passwords after first login through User Management.

## API Reference

### Authentication Functions

**`authenticate(username, password)`**
- Validates credentials
- Returns user data or None

**`add_user(username, password, role, name, email)`**
- Creates new user (admin only)
- Returns True if successful

**`delete_user(username)`**
- Removes user (admin only, except admin)
- Returns True if successful

**`check_permission(required_role)`**
- Checks if current user has permission
- Returns True/False

**`logout()`**
- Ends current session
- Redirects to login

## Notes

- Users are stored in `config/users.json`
- Passwords are hashed and never stored in plain text
- Session persists until logout or browser close
- Default users are recreated if file is deleted

---
**Security Reminder**: Change default passwords in production!
