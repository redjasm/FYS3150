#include <armadillo>
#include <iostream>
#include <fstream>
#include <chrono>

double f(double x)
{
    return 100 * exp(-10 * x);  // Function f(x) as given in the Poisson equation
}

int main()
{
    double N = 1.0;  // Full length of x axis

    for (int I = 1; I <= 7; ++I)
    {
        int n = (int)std::pow(10, I);  // Discretization steps
        double h = N / n;  // Step size
        arma::vec x = arma::linspace(0, N, n + 1);  // Discretized x values
        arma::vec g = arma::vec(n + 1);  // Right-hand side vector g
        arma::vec v = arma::vec(n + 1);  // Solution vector v
        arma::vec scratch = arma::vec(n + 1);  // Scratch space (similar to the C code)

        v(0) = 0;  // Boundary condition at x=0
        v(n) = 0;  // Boundary condition at x=1

        // Initialize g using function f(x) and boundary conditions
        g(1) = f(x(1)) * h * h + v(0);
        g(n - 1) = f(x(n - 1)) * h * h + v(n);
        for (int i = 2; i < n - 1; ++i)
        {
            g(i) = f(x(i)) * h * h;
        }

        // Tridiagonal matrix components: a (subdiagonal), b (main diagonal), c (superdiagonal)
        arma::vec a = arma::vec(n).fill(-1.0);  // Subdiagonal vector a
        arma::vec b = arma::vec(n).fill(2.0);   // Main diagonal vector b
        arma::vec c = arma::vec(n).fill(-1.0);  // Superdiagonal vector c

        // Start measuring time
        auto start = std::chrono::high_resolution_clock::now();

        // Forward substitution step (Thomas algorithm)
        scratch(0) = c(0) / b(0);
        g(0) = g(0) / b(0);

        for (int i = 1; i < n; ++i)
        {
            if (i < n - 1)
            {
                // Update scratch space for the superdiagonal
                scratch(i) = c(i) / (b(i) - a(i) * scratch(i - 1));
            }
            // Update right-hand side vector g
            g(i) = (g(i) - a(i) * g(i - 1)) / (b(i) - a(i) * scratch(i - 1));
        }

        // Backward substitution step (solving for v)
        v(n - 1) = g(n - 1);  // Last element of the solution vector

        for (int i = n - 2; i >= 0; --i)
        {
            // Use scratch values to solve for v
            v(i) = g(i) - scratch(i) * v(i + 1);
        }

        // Stop measuring time
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;

        // Calculate the elapsed time and print it
        double duration_seconds = duration.count();
        std::cout << "Elapsed time for n = " << n << " is " << duration_seconds << " seconds.\n";

        // Write the results to file, including both x and v
        arma::mat A = arma::mat(n + 1, 2);  // Matrix to store x and v for file output
        for (int i = 0; i < n + 1; ++i)
        {
            A(i, 0) = x(i);  // x values
            A(i, 1) = v(i);  // Corresponding v values
        }

        // Create a filename based on the current iteration I and write to file
        std::string filename = "out_e" + std::to_string(I) + ".txt";
        std::ofstream file(filename);
        file << A;
        file.close();
    }

    return 0;
}

// #include <armadillo>
// #include <iostream>
// #include <fstream>
// #include <chrono>

// double f(double x)
// {
//     return 100 * exp(-10 * x);
// }

// int main()
// {
//     double N = 1.0;

//     for (int I = 1; I <= 7; ++I)
//     {
//         int n = (int)std::pow(10, I);
//         double h = N / n;
//         arma::vec x = arma::linspace(0, N, n + 1);
//         arma::vec g = arma::vec(n + 1);
//         arma::vec gt = arma::vec(n + 1); // g-tilde
//         arma::vec v = arma::vec(n + 1);
//         arma::vec vt = arma::vec(n + 1); // v-tilde

//         v(0) = 0;
//         v(n) = 0;

//         g(1) = f(x(1)) * h * h + v(0);
//         g(n - 1) = f(x(n - 1)) * h * h + v(n);
//         for (int i = 2; i < n - 1; ++i)
//         {
//             g(i) = f(x(i)) * h * h;
//         }

//         arma::vec a = arma::vec(n).fill(-1.0);
//         arma::vec b = arma::vec(n).fill(2.0); // b
//         arma::vec bt = arma::vec(n);          // b-tilde
//         arma::vec c = arma::vec(n).fill(-1.0);

//         // Start measuring time
//         auto start = std::chrono::high_resolution_clock::now();

//         bt(0) = b(0);
//         gt(0) = g(0);
//         for (int i = 1; i < n; ++i)
//         {
//             bt(i) = b(i) - (a(i) / bt(i - 1)) * c(i - 1);
//             gt(i) = g(i) - (a(i) / bt(i - 1)) * gt(i - 1);
//         }

//         v(n - 1) = gt(n - 1) / bt(n - 1);

//         for (int i = n - 2; i >= 0; --i)
//         {
//             v(i) = (gt(i) - c(i) * v(i + 1)) / bt(i);
//         }

//         // Stop measuring time
//         auto end = std::chrono::high_resolution_clock::now();
//         std::chrono::duration<double> duration = end - start;

//         // Calculate the elapsed time.
//         double duration_seconds = duration.count();
//         std::cout << "Elapsed time for n = " << n << " is " << duration_seconds << " seconds.\n";

//         // Writing results to file
//         arma::mat A = arma::mat(n + 1, 2);
//         for (int i = 0; i < n + 1; ++i)
//         {
//             A(i, 0) = x(i);
//             A(i, 1) = v(i);
//         }

//         std::string filename = "out_e" + std::to_string(I) + ".txt";
//         std::ofstream file(filename);
//         file << A;
//         file.close();
//     }

//     return 0;
// }
