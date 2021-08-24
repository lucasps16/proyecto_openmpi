#include<stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <fstream> 
#include <vector>
#include <string>
#include <cmath>  
#include <iostream>
#include <sys/time.h>
#include <omp.h>
#include <mpi.h>


using namespace std;


#define THREADS 8
double **mediaTotal= new double*[THREADS];
double **sdTotal= new double*[THREADS];

struct args{
    double *x;
    int rows,columns,id;
};

struct args2{
    double *x;
    int rows,columns,id;
    double *medias;
};
double *sumVectors(double a[], double  b[], int  len) {
 double* c = new double[len];
 for (int i = 0; i < len; i++){
      c[i] = a[i] + b[i];
 }

 return c;
}

double  *subVectors(double  a[], double  b[], int len) {
 double* c = new double[len];
 for (int i = 0; i < len; i++){
      c[i] = a[i] - b[i];
      c[i] = pow(c[i], 2.0);
 }

 return c;
}





void *media (void *input) {

    double *x= ((struct args *)input)->x;
    int columns = ((struct args *)input)->columns;
    int rows = ((struct args *)input)->rows;
    int threadId = ((struct args *)input)->id;
    int initIteration = ((rows)/THREADS) * threadId;
    int endIteration = initIteration + (((rows)/THREADS) );
    double *previous= new double[columns];
    double *sum= new double[columns];


    for (int i= initIteration; i<endIteration ;i++){
        double  arr[columns]; 
        for (int j= 0; j< columns ; j++){

            int index= columns*i + j;
            arr[j]= x[index];
            //cout<<arr[j]<<endl;
            
        }
        previous= sumVectors(previous, arr, columns);

    }

    for (int i=0; i< columns; i++){
        previous[i]=previous[i]/rows;
        //cout<<threadId<< " "<<previous[i]<<endl;

    }
    mediaTotal[threadId]= previous;

    return 0;



}

void *deviation (void *input) {

    double *x= ((struct args2 *)input)->x;
    int columns = ((struct args2 *)input)->columns;
    int rows = ((struct args2 *)input)->rows;
    int threadId = ((struct args2 *)input)->id;
    double *media = ((struct args2 *)input)->medias;

    int initIteration = ((rows)/THREADS) * threadId;
    int endIteration = initIteration + (((rows)/THREADS));
    double *previous= new double[columns];
    double *sum= new double[columns];
    double *sub= new double[columns];



    for (int i= initIteration; i<endIteration ;i++){
        double  arr[columns]; 
        for (int j= 0; j< columns ; j++){
            int index= columns*i + j;
            //cout<<index<<endl;
            arr[j]= x[index];
        }

        //cout<<"termino"<<endl;
        sub= subVectors(arr, media, columns);
        previous= sumVectors(previous, sub, columns);
        //memcpy(&previous, &sum, columns);

    }

    for (int i=0; i< columns; i++){
        previous[i]= previous[i]/rows;
        //printf("deviation %f \n",previous[i]);

    }

    sdTotal[threadId]= previous;

    return 0;



}

double flag(double* x, double* mean, double* sd, int len){

    double *awnsers= new double[len];
    double p=1.0;
    for (int i= 0 ; i<len; i++){
        p= p* (1/(sqrt (2*M_PI*sd[i]))) * exp(-1*(pow(x[i]-mean[i],2.0)/(2*sd[i])));
        
    }

    return p;

}

int main(){


    int size;
    int rank;

    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);


    int rows_training=227834;
    int columns=29;
    int threadId[THREADS],threadId2[THREADS],*retval;
    pthread_t thread[THREADS];
    pthread_t thread2[THREADS];
    double *matriz = new double[rows_training*columns];
    struct timeval tval_before, tval_after, tval_result;
    char buf[64];


    ifstream coeff("x_trainingnew.csv"); 
    if (coeff.is_open()) //if the file is open
	{
		//ignore first line
		string line;


        int i=0;
		while (getline(coeff, line,',')) //while the end of file is NOT reached
		{		 
            double num= stod(line);
            matriz[i]= num;
            i++;
           

        }
	}
	coeff.close(); 

    gettimeofday(&tval_before, NULL);
    double *medias= new double[columns];
    double *deviations= new double[columns];
    
    #pragma omp parallel num_threads(THREADS)
    {
        int ID = omp_get_thread_num();

        struct args *Inputs = (struct args *)malloc(sizeof(struct args));
        Inputs->x =matriz;
        Inputs->rows = rows_training;
        Inputs->columns = columns;
        Inputs->id = ID;
        media((void *)Inputs);
    }

    for(int i = 0; i < THREADS; i++){
        medias = sumVectors(medias, mediaTotal[i], columns); 
    }

    #pragma omp parallel num_threads(THREADS)
    {
        int ID2 = omp_get_thread_num();

        struct args2 *InputsSD = (struct args2 *)malloc(sizeof(struct args2));
        InputsSD->x =matriz;
        InputsSD->rows = rows_training;
        InputsSD->columns = columns;
        InputsSD->medias= medias;
        InputsSD->id = ID2;
        deviation((void *)InputsSD);
    }


    for(int i = 0; i < THREADS; i++){
        deviations = sumVectors(deviations, sdTotal[i], columns); 
    }



    //Calculo de tiempo de ejecucion
    gettimeofday(&tval_after, NULL);
    timersub(&tval_after, &tval_before, &tval_result);
    snprintf(buf, sizeof buf, "%ld.%06ld", tval_result.tv_sec, tval_result.tv_usec);
    string time = (string)(buf);
    string info = "  Threads: " + to_string(THREADS) + " TIME ELAPSED: " + time + "\n";

    //Escritura de informacion de ejecucion en archivo de texto
    ofstream myfile;
    myfile.open("infoOpenMP.txt", ios_base::app);
    if (myfile.is_open())
    {
        myfile << info;
        myfile.close();
    }

    for(int i = 0; i < columns; i++){
        printf("media %f\n",medias[i]);
        
        
    }

    for(int i = 0; i < columns; i++){
        printf("deviation %f\n",deviations[i]);
        
        
    }

 


    //TESTING

    /*
    string line2;
    ifstream testing("x_testingnew.csv"); 

    int rows_testing= 56958;


    for (int i=0; i<rows_testing; i++){
        double *x = new double[columns];
        for (int j=0; j<columns; j++){
            getline(testing, line2,',');
            //cout<<line2<<endl;
            double num= stod(line2);
            x[j]= num;
            //cout<<x[j]<<endl;
        }
        double p= flag(x,valor1,valor2,columns);
        printf("p %.10f \n",p);
    }
    
    testing.close();
    */



}
