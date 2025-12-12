import javax.swing.*;
import java.awt.*;
import java.sql.*;

public class ReservationForm extends JFrame {
    private String username;
    private JComboBox<String> trainList;
    private JTextField passengerName;
    private JButton bookBtn;

    public ReservationForm(String username) {
        this.username = username;

        setTitle("Book Ticket - " + username);
        setSize(400, 200);
        setLayout(new GridLayout(3, 2));
        setDefaultCloseOperation(EXIT_ON_CLOSE);

        add(new JLabel("Select Train:"));
        trainList = new JComboBox<>();
        loadTrains();
        add(trainList);

        add(new JLabel("Passenger Name:"));
        passengerName = new JTextField();
        add(passengerName);

        bookBtn = new JButton("Book Ticket");
        add(bookBtn);

        bookBtn.addActionListener(e -> bookTicket());

        JButton cancelBtn = new JButton("Cancel Ticket");
        cancelBtn.addActionListener(e -> new CancellationForm(username));
        add(cancelBtn);

        JButton viewBtn = new JButton("View My Bookings");
        viewBtn.addActionListener(e -> new ViewBookings(username));
        add(viewBtn);

        setVisible(true);
    }

    private void loadTrains() {
        try (Connection conn = DatabaseConnection.getConnection()) {
            String sql = "SELECT train_name FROM trains";
            PreparedStatement stmt = conn.prepareStatement(sql);
            ResultSet rs = stmt.executeQuery();
            while (rs.next()) {
                trainList.addItem(rs.getString("train_name"));
            }
        } catch (SQLException e) {
            e.printStackTrace();
        }
    }

    private void bookTicket() {
        String train = (String) trainList.getSelectedItem();
        String passenger = passengerName.getText();
        String pnr = Utils.generatePNR();

        try (Connection conn = DatabaseConnection.getConnection()) {
            String sql = "INSERT INTO bookings (username, train_name, passenger_name, pnr) VALUES (?, ?, ?, ?)";
            PreparedStatement stmt = conn.prepareStatement(sql);
            stmt.setString(1, username);
            stmt.setString(2, train);
            stmt.setString(3, passenger);
            stmt.setString(4, pnr);
            stmt.executeUpdate();

            JOptionPane.showMessageDialog(this, "Ticket booked! PNR: " + pnr);
        } catch (SQLException e) {
            e.printStackTrace();
        }
    }
}
