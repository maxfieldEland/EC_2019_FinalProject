// GEP.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
// ge_dep_hunan_model.cpp : Defines the entry point for the console application.
//gep.cpp : Defines the entry point for the console application.
// To evolve crowd model simulation.
#include "stdafx.h"
#include "stdlib.h"
#include "stdio.h"
#include "time.h"
#include "math.h"
#include "string.h"

#include "windows.h"

#define MAXGENS	100
#define H	5
#define T	(H+1)
#define NVARS (H+T+T)
#define POPSIZE	20
int generation;
int terminate_num = 3;	//{d, w, n, c}
int function_num = 4;	//{+,-,*,/}
int init_flag;

HANDLE	flag_mutex;	

#define LBOUND -10
#define UBOUND 10

typedef struct
{
	double	gene[NVARS];   //gene[0] for position, and gene[1] for order
	int		constant_num;
	double	f;
}CHROMOSOME;

CHROMOSOME population[POPSIZE+1], newpopulation[POPSIZE+1];


#define MAXEVALS 20001
double avgf[MAXEVALS + 1];
double fbest;
int evals;
int run;
double randval(double a, double b)
{
	return a + (b - a) * rand() /(double) RAND_MAX;
}

//========================================================
//structure to represent components.  <type, value, dvalue, L, R>
//type: component type, e.g., terminate or operator
//value: component id;
//dvalue: if the component is constant, then this is the constant value.
//value = {function， terminate}
struct LINK_COMP
{
	int value;
	double dvalue;
	struct LINK_COMP *left;
	struct LINK_COMP *right;
};
struct LINK_COMP *link_root[POPSIZE+1], link_comp[POPSIZE+1][NVARS];
int constant_index;

FILE *outf[POPSIZE+1];
FILE *outf_str[POPSIZE+1];
void output_rule(const struct LINK_COMP * node, FILE *f_str)
{
	int i, j, k;
	if(node == NULL) return;
	if(node->value >= function_num){  //terminate value
		switch(node->value - function_num){
		case 0: fprintf(f_str,"D");	break;
		case 1: fprintf(f_str,"W");	break;
		case 2: fprintf(f_str,"N");	break;			
		case 3: fprintf(f_str,"%g", node->dvalue);	break;
		}
	}else{	
		fprintf(f_str, "(");
		output_rule(node->left, f_str);
		switch(node->value){
		case 0:fprintf(f_str, " + ");	break;
		case 1:fprintf(f_str, " - ");	break;
		case 2:fprintf(f_str, " * ");	break;
		case 3:fprintf(f_str, " $ ");	break;
		}

		output_rule(node->right,f_str);
		fprintf(f_str, ")");
	}
}


//decode gene expression to link format.
//decode population[i]
void decode_gene(int I)
{
	double * rule;
	int op = -1, i = 0, k = 0;
 	CHROMOSOME * p;
	if(init_flag == 0) p = &population[I];
	else p = &newpopulation[I];

	rule = p->gene;
	for(i = 0; i < NVARS - T; i++){
		link_comp[I][i].dvalue = 0;
		link_comp[I][i].value = (int) rule[i];
		link_comp[I][i].left = link_comp[I][i].right = NULL;
	}

	op = -1, i = 1;
	link_root[I] = &link_comp[I][0];
	p->constant_num = 0;
	if(link_root[I]->value < function_num){  //start with an operator.
		do{ 	
			//if start, find the first operator, otherwise, find the next operator
			do{op++; if(op > i)break;}while(link_comp[I][op].value >= function_num);

			//the tree end.
			if(op > i) break;
			//if this is an operator, then assign its left and right pointer.  [wide-prefer-travel]
			if(link_comp[I][op].value < function_num){
				
				if(i >= NVARS - 1){ i = -1; break;}
				
				link_comp[I][op].left = &link_comp[I][i];	
				if(link_comp[I][i].value == function_num + terminate_num - 1){ //constant
					link_comp[I][i].dvalue = rule[H+T+p->constant_num];
					p->constant_num++;
				}
				i++;

				//each operator always have two parameters				
				link_comp[I][op].right = &link_comp[I][i];
				if(link_comp[I][i].value == function_num + terminate_num - 1){ //constant
					link_comp[I][i].dvalue = rule[H+T+p->constant_num];
					p->constant_num++;
				}
				i++;			
			}
		}while(true);
		if(op < i  &&i == -1){ printf("ERROR RULE111"); getchar();}
	}else printf("terminate000000\t");	
}

void output_gene(int ip)
{		
	int op = -1, i = 0, k = 0, j;	
	char name[100];
	const CHROMOSOME * p;
	if(init_flag == 0) p = & population[ip];
	else p = &newpopulation[ip];
	decode_gene(ip);
	if(ip < POPSIZE){
		sprintf_s(name,"output_%d.txt", ip);
		fopen_s(&outf[ip], name,"w");	
		sprintf_s(name,"output_%d_str.txt", ip);
		fopen_s(&outf_str[ip], name, "w");
	}else{
		sprintf_s(name,"output_%d_%d_str.txt", ip, run);
		fopen_s(&outf_str[ip], name, "w");
		output_rule(link_root[ip], outf_str[ip]);
		fprintf(outf_str[ip],"\n");		
		fclose(outf_str[ip]);

		sprintf_s(name,"output_%d_%d.txt", ip, run);
		fopen_s(&outf[ip], name,"w");	
	}
	fprintf(outf[ip],"%d\n", H);	
	for(i = 0; i < NVARS; i++) fprintf(outf[ip], "%g\t", p->gene[i]);
	fprintf(outf[ip], "\n");
	fclose(outf[ip]);
	
}



HANDLE	tid[POPSIZE];
DWORD	ID[POPSIZE];
int		run_flag[POPSIZE];
int		ready_flag[POPSIZE];
void syn()
{
	int i, j;
	int flag, count = 0, v[POPSIZE];
	do{
		flag = 0;	
		for(i = 0; i < POPSIZE; i++){
			v[i] = run_flag[i];
		}	
		for(i = 0; i < POPSIZE; i++){
			if(v[i] == 1){
				flag = 1;
				break;	 	
			}
		}		
		::Sleep(50);
	}while(flag == 1);
}

	

bool read_signal(int I)
{
	int i, j;
	int flag, count = 0, v[POPSIZE];
	do{
		if(ready_flag[I] == 1){ready_flag[I] = 0;break;}		
		::Sleep(50);
	}while(true);
	return true;	
}

DWORD WINAPI thread (PVOID ID)
{
	char name[100];
	int * pid = (int *) ID;
	int ip = *pid;
	int i, j;
	double result;
	CHROMOSOME *p;
	FILE *f;
	DWORD d;
	while(read_signal(ip)){	
		if(init_flag == 0) p = & population[ip];
		else p = &newpopulation[ip];

		d = WaitForSingleObject(flag_mutex, INFINITE);  
		output_gene(ip);
		::ReleaseMutex(flag_mutex);

		Sleep(200);

		sprintf_s(name,"java -jar jinghui.jar %d 0", ip);		
		int ret = system(name);
		if(ret == 1) printf("something error happen");
		
		d = WaitForSingleObject(flag_mutex, INFINITE);  
		sprintf_s(name,"output_%d.txt",ip);		
		fopen_s(&f, name,"r");		
		fscanf_s(f,"%lf\n", &result);
		fclose(f);
		p->f = result;
		//printf("%d\t%g\n", ip, p->f);
		
		run_flag[ip] = 0;
		::ReleaseMutex(flag_mutex);
	}
	return 1;
}

void create_thread()
{
	int i, j;
	DWORD v;
	for(i = 0; i < POPSIZE; i++) {	
		ready_flag[i] = 0;
		ID[i] = i;			
		tid[i] = CreateThread(NULL, 0, thread, &ID[i], 0, &v);			
	}
	flag_mutex = ::CreateMutex(NULL,FALSE,NULL);
}


void initialize()
{
	int i, j, k;
	int ibest = 0;
	evals = 0;
	fbest = 1e10;
	for(i = 0; i < POPSIZE; i++){	
		for(k = 0; k < NVARS; k++){
			if(k < H){
				if(k == 0 || randval(0,1) < 0.5) population[i].gene[k] = rand() % function_num;
				else population[i].gene[k] = function_num + rand()%terminate_num;
			}else if(k < H + T){
				population[i].gene[k] = function_num + rand()%terminate_num;
			}else population[i].gene[k] = randval(LBOUND, UBOUND);
		}	
		run_flag[i] = 1; ready_flag[i] = 1;
	}
	syn();

	
	for(i = 0; i < POPSIZE; i++){	
		if(population[i].f < population[ibest].f) ibest = i;

		if(population[i].f < fbest) fbest = population[i].f;
		avgf[evals] += fbest;
		evals++;
	}
	population[POPSIZE] = population[ibest];
}


//function£ºgenerate a randomv with N(0,1) distribution 
static int phase = 0;
double gaussian()                                                       
{
    static double V1, V2, S; 
    double X;    
    if ( phase == 0 ) {
        do {
            double U1 = (double)rand() /(double)RAND_MAX;
            double U2 = (double)rand() /(double)RAND_MAX;
            V1 = 2 * U1 - 1;
            V2 = 2 * U2 - 1;
            S = V1 * V1 + V2 * V2;
        } while(S >= 1 || S == 0);
        
        X = V1 * sqrt(-2 * log(S) / S);
    } else
        X = V2 * sqrt(-2 * log(S) / S);
    return X;
        
    phase = 1 - phase;
}



double gauss(double a, double b)
{
	return a + gaussian() * b;
}

double cauchy(double location, double t)
{
	double v1 = gauss(0,1);
	double v2 = gauss(0,1);
	if (v2 != 0)
	{
		return t * v1 / v2 + location;
	}
	return location;
}

void production()
{
	int i, j, k, r1, r2, r3,r4, r5;
	int Q_index;
	double F, CR;
	for(i = 0; i < POPSIZE; i++){
		newpopulation[i] = population[i];
		F = randval(0,1);
		CR = randval(0,1);
		r1 = rand() % POPSIZE;
		do{r2 = rand() % POPSIZE;}while(r2 == r1);
		do{r3 = rand() % POPSIZE;}while(r3 == r2 || r3 == r1);
		do{r4 = rand() % POPSIZE;}while(r4 == r2 || r4 == r1 || r4 == r3);
		do{r5 = rand() % POPSIZE;}while(r5 == r2 || r5 == r1 || r5 == r4 || r5 == r3);
		k = rand() % (NVARS);		
		for(j = 0; j < NVARS; j++){
			if(randval(0,1) < CR || (k == j)){
				newpopulation[i].gene[j] =  (int) (population[i].gene[j] + F * ( population[POPSIZE].gene[j] -  population[i].gene[j]) +  cauchy(0, 1) *(population[r1].gene[j] -  population[r2].gene[j]));

					if(j < H){
						while(newpopulation[i].gene[j] < 0 || newpopulation[i].gene[j] >= function_num + terminate_num ||
							(j == 0 && newpopulation[i].gene[j] >= function_num)) 
							newpopulation[i].gene[j] = rand() % (function_num + terminate_num);
					}else if(j < H + T){
						if(newpopulation[i].gene[j] < function_num || newpopulation[i].gene[j] >= function_num + terminate_num)
							newpopulation[i].gene[j] = function_num + rand() % terminate_num;
					}else{ 
						//without bound
						if(newpopulation[i].gene[j] < LBOUND || newpopulation[i].gene[j] > LBOUND) 
							newpopulation[i].gene[j] = randval(LBOUND, UBOUND);
					}
			}else newpopulation[i].gene[j] = population[i].gene[j];
		}
		run_flag[i] = 1; ready_flag[i] = 1;
	}
	syn();

	for(i = 0; i < POPSIZE; i++){		
		if(newpopulation[i].f < population[i].f) population[i] = newpopulation[i];
		if(population[i].f < population[POPSIZE].f) population[POPSIZE] = population[i];

		if(population[i].f < fbest) fbest = population[i].f;
		avgf[evals] += fbest;
		evals++;
	}	
}


void DE()
{
	printf("go init...\n");
	init_flag = 0;
	initialize();
	init_flag = 1;
	double gen_fitness[100];
	char name[100];
	sprintf(name,"%d_fitness.txt", run);
	FILE *f = fopen(name,"w");
	fprintf(f, "%g\n", population[POPSIZE].f);
	for(generation = 1; generation <= 100; generation++){		
		production();		
		printf("========================\n%d\t%d\t%d\t%g\n", run, evals, generation,  population[POPSIZE].f);	
		newpopulation[POPSIZE] = population[POPSIZE];
		output_gene(POPSIZE);
		gen_fitness[generation] = population[POPSIZE].f;	
		fprintf(f, "%g\n", population[POPSIZE].f);
	}	
	fclose(f);
}

int main()
{
	srand(time(NULL));
	printf("start de...\n");
	for(evals = 0; evals < 10000; evals++) avgf[evals] = 0;
	FILE *f;
	create_thread(); 
	for(run = 20; run < 30; run++){
		DE();
		fopen_s(&f, "results.txt","a");
		fprintf(f,"%d\t%g\n", run, population[POPSIZE].f);
		fclose(f);
	}
	fopen_s(&f, "avg_fitness.txt","w");
	for(evals = 0; evals < 10000; evals++){
		fprintf(f,"%d\t%g\n", evals, avgf[evals]/1.);
	}
	fclose(f);
	return 0;
}