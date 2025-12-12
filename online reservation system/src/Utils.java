import java.util.Random;

public class Utils {
    public static String generatePNR() {
        Random rand = new Random();
        return "PNR" + (100000 + rand.nextInt(900000));
    }
}
