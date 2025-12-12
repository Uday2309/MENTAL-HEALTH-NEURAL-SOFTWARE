import javax.swing.*;
import java.awt.*;
import java.sql.*;

public class CancellationForm extends JFrame {
    private String username;
    private JTextField pnrField;

    public CancellationForm(String username) {
        this.username = username;

        setTitle("Cancel Ticket");
        setSize(300, 150);
        setLayout(new GridLayout(2, 2));

        add(new JLabel("Enter PNR:"));
        pnrField = new JTextField();
        add(pnrField);

        JButton cancelBtn = new JButton("Cancel Ticket");
        cancelBtn.addActionListener(e -> cancelTicket());
        add(cancelBtn);

        setVisible(true);
    }

    private void cancelTicket() {
        String pnr = pnrField.getText();

        try (Connection conn = DatabaseConnection.getConnection()) {
            String sql = "DELETE FROM bookings WHERE pnr=? AND username=?";
            PreparedStatement stmt = conn.prepareStatement(sql);
            stmt.setString(1, pnr);
            stmt.setString(2, username);
            int rows = stmt.executeUpdate();

            if (rows > 0) {
                JOptionPane.showMessageDialog(this, "Ticket cancelled successfully!");
            } else {
                JOptionPane.showMessageDialog(this, "PNR not found or unauthorized!");
            }
        } catch (SQLException e) {
            e.printStackTrace();
        }
    }
}
