import javax.swing.*;
import java.awt.*;
import java.sql.*;

public class ViewBookings extends JFrame {
    private String username;
    private JTextArea bookingArea;

    public ViewBookings(String username) {
        this.username = username;

        setTitle("My Bookings");
        setSize(400, 300);
        bookingArea = new JTextArea();
        bookingArea.setEditable(false);
        add(new JScrollPane(bookingArea), BorderLayout.CENTER);

        loadBookings();

        setVisible(true);
    }

    private void loadBookings() {
        try (Connection conn = DatabaseConnection.getConnection()) {
            String sql = "SELECT * FROM bookings WHERE username=?";
            PreparedStatement stmt = conn.prepareStatement(sql);
            stmt.setString(1, username);
            ResultSet rs = stmt.executeQuery();

            StringBuilder sb = new StringBuilder();
            while (rs.next()) {
                sb.append("PNR: ").append(rs.getString("pnr"))
                  .append(" | Train: ").append(rs.getString("train_name"))
                  .append(" | Passenger: ").append(rs.getString("passenger_name"))
                  .append("\n");
            }
            bookingArea.setText(sb.toString());
        } catch (SQLException e) {
            e.printStackTrace();
        }
    }
}
