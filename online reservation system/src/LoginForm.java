import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;

public class LoginForm extends JFrame {
    private JTextField usernameField;
    private JPasswordField passwordField;

    public LoginForm() {
        setTitle("Login - Online Reservation System");
        setSize(350, 200);
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setLocationRelativeTo(null);

        JPanel panel = new JPanel(new GridLayout(3, 2, 10, 10));

        panel.add(new JLabel("Username:"));
        usernameField = new JTextField();
        panel.add(usernameField);

        panel.add(new JLabel("Password:"));
        passwordField = new JPasswordField();
        panel.add(passwordField);

        JButton loginButton = new JButton("Login");
        loginButton.addActionListener(new LoginAction());
        panel.add(loginButton);

        JButton registerButton = new JButton("Register");
        registerButton.addActionListener(new RegisterAction());
        panel.add(registerButton);

        add(panel);
        setVisible(true);
    }

    // Action for login
    private class LoginAction implements ActionListener {
        @Override
        public void actionPerformed(ActionEvent e) {
            String username = usernameField.getText().trim();
            String password = new String(passwordField.getPassword()).trim();

            if (username.isEmpty() || password.isEmpty()) {
                JOptionPane.showMessageDialog(LoginForm.this, "Please fill in all fields.");
                return;
            }

            try (Connection conn = DatabaseConnection.getConnection()) {
                String query = "SELECT role FROM users WHERE username = ? AND password = ?";
                PreparedStatement stmt = conn.prepareStatement(query);
                stmt.setString(1, username);
                stmt.setString(2, password);
                ResultSet rs = stmt.executeQuery();

                if (rs.next()) {
                    String role = rs.getString("role");
                    JOptionPane.showMessageDialog(LoginForm.this, "Login successful! Role: " + role);

                    // Open respective dashboard
                    dispose();
                    if (role.equalsIgnoreCase("admin")) {
                        new AdminTrainManagement();
                    } else {
                        new ReservationForm(username); // Pass username here ✅
                    }
                } else {
                    JOptionPane.showMessageDialog(LoginForm.this, "Invalid username or password.");
                }

            } catch (SQLException ex) {
                JOptionPane.showMessageDialog(LoginForm.this, "Database error: " + ex.getMessage());
            }
        }
    }

    // Action for register
    private class RegisterAction implements ActionListener {
        @Override
        public void actionPerformed(ActionEvent e) {
            String username = usernameField.getText().trim();
            String password = new String(passwordField.getPassword()).trim();

            if (username.isEmpty() || password.isEmpty()) {
                JOptionPane.showMessageDialog(LoginForm.this, "Please fill in all fields.");
                return;
            }

            try (Connection conn = DatabaseConnection.getConnection()) {
                String query = "INSERT INTO users (username, password, role) VALUES (?, ?, 'user')";
                PreparedStatement stmt = conn.prepareStatement(query);
                stmt.setString(1, username);
                stmt.setString(2, password);
                stmt.executeUpdate();

                JOptionPane.showMessageDialog(LoginForm.this, "User registered successfully!");

            } catch (SQLException ex) {
                JOptionPane.showMessageDialog(LoginForm.this, "Error: " + ex.getMessage());
            }
        }
    }
}
