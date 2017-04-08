/*
  Branch and bound algorithm to find the minimum of continuous binary 
  functions using interval arithmetic.

  Sequential version

  Author: Frederic Goualard <Frederic.Goualard@univ-nantes.fr>
  v. 1.0, 2013-02-15
*/

// POUR EXÉCUTER : mpirun -np 2 --hostfile ./hostfile ./optimization-par

#include <iostream>
#include <iterator>
#include <string>
#include <stdexcept>
#include "interval.h"
#include "functions.h"
#include "minimizer.h"
#include "mpi.h"
#include "omp.h"
#include <ctime>

using namespace std;


// Split a 2D box into four subboxes by splitting each dimension
// into two equal subparts
void split_box(const interval& x, const interval& y,
	       interval &xl, interval& xr, interval& yl, interval& yr)
{
	double xm = x.mid();
	double ym = y.mid();
	xl = interval(x.left(),xm);
	xr = interval(xm,x.right());
	yl = interval(y.left(),ym);
	yr = interval(ym,y.right());
}

// Branch-and-bound minimization algorithm
void minimize(itvfun f,  // Function to minimize
	      const interval& x, // Current bounds for 1st dimension
	      const interval& y, // Current bounds for 2nd dimension
	      double threshold,  // Threshold at which we should stop splitting
	      double& min_ub,  // Current minimum upper bound
	      minimizer_list& ml) // List of current minimizers
{
	interval fxy = f(x,y);

	if (fxy.left() > min_ub) { // Current box cannot contain minimum?
		return ;
	}

	if (fxy.right() < min_ub) { // Current box contains a new minimum?
		#pragma omp critical
		{
			min_ub = fxy.right();
			// Discarding all saved boxes whose minimum lower bound is 
			// greater than the new minimum upper bound
			auto discard_begin = ml.lower_bound(minimizer{0,0,min_ub,0});
			ml.erase(discard_begin,ml.end());
		}
	}
	
	// Checking whether the input box is small enough to stop searching.
	// We can consider the width of one dimension only since a box
	// is always split equally along both dimensions
	if (x.width() <= threshold) { 
		// We have potentially a new minimizer
		#pragma omp critical
			ml.insert(minimizer{x,y,fxy.left(),fxy.right()});
		
		return ;
	}

	// The box is still large enough => we split it into 4 sub-boxes
	// and recursively explore them
	interval xl, xr, yl, yr;
	split_box(x,y,xl,xr,yl,yr);

	omp_set_num_threads(4);
	
	#pragma omp parallel
	#pragma omp sections
	{
		#pragma omp section
		minimize(f,xl,yl,threshold,min_ub,ml);
		#pragma omp section
		minimize(f,xl,yr,threshold,min_ub,ml);
		#pragma omp section
		minimize(f,xr,yl,threshold,min_ub,ml);
		#pragma omp section
		minimize(f,xr,yr,threshold,min_ub,ml);
	}
}


int main(int argc, char * argv[])
{
	int rank, numProcs;

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	cout.precision(16);
	
	// By default, the currently known upper bound for the minimizer is +oo
	double min_ub = numeric_limits<double>::infinity();
	double local_min_ub = numeric_limits<double>::infinity();
	
	// List of potential minimizers. They may be removed from the list
	// if we later discover that their smallest minimum possible is 
	// greater than the new current upper bound
	minimizer_list minimums;
	
	// Threshold at which we should stop splitting a box
	double precision;

	// Name of the function to optimize
	string choice_fun;

	// The information on the function chosen (pointer and initial box)
	opt_fun_t fun;
	
	// Execution time
	time_t debut, fin;
	
	// Array of sub-intervals
	interval tabX[numProcs];
	interval tabY[numProcs];
	
	// Slice which contains one sub-interval of X
	interval sliceX[1];
	
	// Size of sub-intervals
	double siX = 0.0;
	double siY = 0.0;
	
	// Left bound of each sub-interval
	double lx = 0.0;
	double ly = 0.0;

	bool good_choice;

	if(rank == 0){
		
		// Asking the user for the name of the function to optimize
		do {
			good_choice = true;

			cout << "Which function to optimize?\n";
			cout << "Possible choices: ";
			
			for (auto fname : functions) {
				cout << fname.first << " ";
			}
			
			cout << endl;
			cin >> choice_fun;

			try {
				fun = functions.at(choice_fun);
			} catch (out_of_range) {
				cerr << "Bad choice" << endl;
				good_choice = false;
			}
		} while(!good_choice);

		// Asking for the threshold below which a box is not split further
		cout << "Precision? ";
		cin >> precision;
		
		// Calcul de la taille des sous-intervalles de X
		siX = (fun.x.width()) / numProcs;
		
		// Borne gauche de l'intervalle de X
		lx = fun.x.left();
		
		// Remplissage des tableaux de sous-intervalles de X
		#pragma omp parallel for
		for (int i = 0 ; i < numProcs ; ++i){	
			tabX[i] = interval(lx + siX * i, lx + siX * (i + 1));
		}
	}
	
	debut = clock();
	
	// Envoi de la fonction et de la précision à toutes les autres machines depuis le rank 0
	MPI_Bcast(&fun, sizeof(opt_fun_t), MPI_BYTE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&precision, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	
	// Envoi des sous-intervalles de X aux autres machines.
	MPI_Scatter(&tabX, sizeof(interval), MPI_BYTE, &sliceX, sizeof(interval), MPI_BYTE, 0, MPI_COMM_WORLD);
	
	// PROBLEME : On ne peut utiliser que des puissances de 2 pour le nombre de machine...
	
	/*
	 * Calcul des sous-intervalles de Y pour obtenir des cubes lors de l'appel à minimize. Chaque machine doit
	 * connaître tous les intervalles de Y afin de vérifier tous les couples (x,y) possibles. De plus, pas besoin
	 * d'envoyer les sous-intervalles de Y car chaque machine est capable de le calculer.
	 * (Note : On ne peut donc pas prendre en compte l'intervalle complet de Y lors de l'appel à minimize, sinon 
	 * on n'obtient pas de cubes)
	 */
	
	// Calcul de la taille des sous-intervalles de Y
	siY = (fun.y.width()) / numProcs;		

	// Borne gauche de l'intervalle de Y
	ly = fun.y.left();
	
	// Remplissage des tableaux de sous-intervalles de Y
	#pragma omp parallel for
	for (int i = 0 ; i < numProcs ; ++i){		
		tabY[i] = interval(ly + siY * i, ly + siY * (i + 1));
	}

	// Calcul du minimum pour chaque sous-intervalle de Y et pour l'intervalle de X de la machine courante
	//#pragma omp parallel
	//{
		//double local_min_ub_private = numeric_limits<double>::infinity();
		//minimizer_list minimums_private;
		
		//#pragma omp parallel for //reduction (min:local_min_ub)
		//for(int i = 0 ; i < numProcs ; ++i){
			//minimize(fun.f,sliceX[0],tabY[i],precision,local_min_ub,minimums);
			
			//#pragma omp critical
			//{
			//	if(local_min_ub > local_min_ub_private)	
			//		local_min_ub = local_min_ub_private;
				
			//	minimums = minimums_private;
			//}
		//}
	//}

	#pragma omp parallel for shared(local_min_ub)
	for(int i = 0 ; i < numProcs ; ++i){
		minimize(fun.f,sliceX[0],tabY[i],precision,local_min_ub,minimums);
	}

	//Note : le reduction(min:local_min_ub) diminue les performances...
	
	// Trouver le minimum des minimum
	MPI_Reduce(&local_min_ub, &min_ub, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
	
	fin = clock();

	// Displaying all potential minimizers
	if(rank == 0){
		/*copy(minimums.begin(),minimums.end(),
		   ostream_iterator<minimizer>(cout,"\n"));*/
		   
		//cout << "Number of minimizers: " << minimums.size() << endl;
		cout << "Upper bound for minimum: " << min_ub << endl;
		cout << "Time : " << (double) (fin - debut) / CLOCKS_PER_SEC << "s" << endl;
	}

	MPI_Finalize();
}
