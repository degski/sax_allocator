
// MIT License
//
// Copyright (c) 2020 degski
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR Allocator PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "include/plf_colony_allocator.hpp"
#include "include/win_virtual_allocator.hpp"
#include "include/mmap_virtual_allocator.hpp"

#include <atomic>
#include <sax/iostream.hpp>
#include <set>

/*
    -fsanitize = address

    C:\Program Files\LLVM\lib\clang\10.0.0\lib\windows\clang_rt.asan_cxx-x86_64.lib
    C:\Program Files\LLVM\lib\clang\10.0.0\lib\windows\clang_rt.asan-preinit-x86_64.lib
    C:\Program Files\LLVM\lib\clang\10.0.0\lib\windows\clang_rt.asan-x86_64.lib

    C:\Program Files (x86)\IntelSWTools\compilers_and_libraries\windows\tbb\lib\intel64_win\vc_mt\tbb.lib
*/

#include <sax/prng_sfc.hpp>
#include <sax/uniform_int_distribution.hpp>

#include <plf/plf_nanotimer.h>

#if defined( NDEBUG )
#    define RANDOM 1
#else
#    define RANDOM 0
#endif

namespace ThreadID {
// Creates a new ID.
[[nodiscard]] inline int get ( bool ) noexcept {
    static std::atomic<int> global_id = 0;
    return global_id++;
}
// Returns ID of this thread.
[[nodiscard]] inline int get ( ) noexcept {
    static thread_local int thread_local_id = get ( false );
    return thread_local_id;
}
} // namespace ThreadID

namespace Rng {
// Chris Doty-Humphrey's Small Fast Chaotic Prng.
[[nodiscard]] inline sax::Rng & generator ( ) noexcept {
    if constexpr ( RANDOM ) {
        static thread_local sax::Rng generator ( sax::os_seed ( ), sax::os_seed ( ), sax::os_seed ( ), sax::os_seed ( ) );
        return generator;
    }
    else {
        static thread_local sax::Rng generator ( sax::fixed_seed ( ) + ThreadID::get ( ) );
        return generator;
    }
}
} // namespace Rng

#undef RANDOM

sax::Rng & rng = Rng::generator ( );

template<typename T, typename C = std::less<T>>
using mi_plf_set = std::set<T, C, mi_colony_node_allocator<T>>;

template<typename T, typename C = std::less<T>>
using stl_plf_set = std::set<T, C, stl_colony_node_allocator<T>>;

template<typename K, typename V>
struct kv_pair {
    K key;
    V value;
};

int main ( ) {

    constexpr std::size_t N = 10'000;

    {
        constexpr std::size_t SegmentSize = 65'536, Capacity = 8 * SegmentSize;

        std::vector<std::size_t, win_allocator<std::size_t, SegmentSize, Capacity>> vctr;

        vctr.reserve ( Capacity );
    }

    /*

    {
        constexpr std::size_t SegmentSize = 65'536, Capacity = 8 * SegmentSize;

        std::vector<std::size_t, win_allocator<std::size_t, SegmentSize, Capacity>> vctr;

        std::cout << "vec constructed" << nl;

        vctr.reserve ( Capacity );

        std::cout << "vec reserved" << nl;

        vctr.emplace_back ( 123 );

        std::cout << "vec allocated" << nl;

        exit ( 0 );

        std::size_t result = 1;

        std::uint64_t duration;
        plf::nanotimer timer;
        timer.start ( );

        for ( std::size_t i = 0; i < N; ++i )
            result = vctr.emplace_back ( i );

        duration = static_cast<std::uint64_t> ( timer.get_elapsed_us ( ) );
        std::cout << std::dec << duration << " us " << result << nl;
    }

    {
        std::set<std::size_t> set;
        std::size_t result = 1;

        std::uint64_t duration;
        plf::nanotimer timer;
        timer.start ( );

        for ( std::size_t i = 0; i < N; ++i )
            result = *set.emplace ( i ).first;

        duration = static_cast<std::uint64_t> ( timer.get_elapsed_us ( ) );
        std::cout << std::dec << duration << " us " << result << nl;
    }

    {
        stl_plf_set<std::size_t> set;
        std::size_t result = 1;

        std::uint64_t duration;
        plf::nanotimer timer;
        timer.start ( );

        for ( std::size_t i = 0; i < N; ++i )
            result = *set.emplace ( i ).first;

        duration = static_cast<std::uint64_t> ( timer.get_elapsed_us ( ) );
        std::cout << std::dec << duration << " us " << result << nl;
    }

    {
        mi_plf_set<std::size_t> set;
        std::size_t result = 1;

        std::uint64_t duration;
        plf::nanotimer timer;
        timer.start ( );

        for ( std::size_t i = 0; i < N; ++i )
            result = *set.emplace ( i ).first;

        duration = static_cast<std::uint64_t> ( timer.get_elapsed_us ( ) );
        std::cout << std::dec << duration << " us " << result << nl;
    }
    */
    return EXIT_SUCCESS;
}

template<typename Type>
struct Foo {
    Type m_member;
};

template<template<typename Type> class TemplateType>
struct Bar {
    TemplateType<int> m_ints;
};

template<template<template<typename> class> class TemplateTemplateType>
struct Baz {
    TemplateTemplateType<Foo> m_foos;
};

using Example = Baz<Bar>;

int main980780 ( ) {

    Example e;

    e.m_foos.m_ints.m_member = 42;
    std::cout << e.m_foos.m_ints.m_member << nl;

    return EXIT_SUCCESS;
}

#ifdef USE_TBB_ALLOCATOR
#    include <tbb/scalable_allocator.h>
#    pragma comment( lib, "tbbmalloc.lib" )
#else
#    define scalable_aligned_malloc _aligned_malloc
#    define scalable_aligned_free _aligned_free
#endif

#if 0

/*----------------------------------------------------------------------
	This file contains a simulation of the cart and pole dynamic system and
 a procedure for learning to balance the pole.	Both are described in
 Barto, Sutton, and Anderson, "Neuronlike Adaptive Elements That Can Solve
 Difficult Learning Control Problems," IEEE Trans. Syst., Man, Cybern.,
 Vol. SMC-13, pp. 834--846, Sept.--Oct. 1983, and in Sutton, "Temporal
 Aspects of Credit Assignment in Reinforcement Learning", PhD
 Dissertation, Department of Computer and Information Science, University
 of Massachusetts, Amherst, 1984.  The following routines are included:

	   main:			  controls simulation interations and implements
						  the learning system.

	   cart_and_pole:	  the cart and pole dynamics; given action and
						  current state, estimates next state

	   get_box:			  The cart-pole's state space is divided into 162
						  boxes.  get_box returns the index of the box into
						  which the current state appears.

 These routines were written by Rich Sutton and Chuck Anderson.	 Claude Sammut
 translated parts from Fortran to C.  Please address correspondence to
 Rich at

		GTE Laboratories Incorporated
		40 Sylvan Road
		Waltham, MA	 02254

 or send email to	  sutton@gte.com   or	anderson@cs.colostate.edu
---------------------------------------
Changes:
  1/93: A bug was found and fixed in the state -> box mapping which resulted
		in array addressing outside the range of the array.	 It's amazing this
		program worked at all before this bug was fixed.  -RSS
----------------------------------------------------------------------*/

#    include <stdlib.h>
#    include <stdio.h>
#    include <imath.h>

#    include <librandom.h>


int get_box(float x, float x_dot, float theta, float theta_dot);
void cart_pole(int action, float *x, float *x_dot, float *theta, float *theta_dot);

#    define prob_push_right( s ) ( 1.0f / ( 1.0f + expf ( -fmaxf ( -50.0f, fminf ( s, 50.0f ) ) ) ) )

#    define N_BOXES 162  /* Number of disjoint boxes of state space. */
#    define ALPHA 1000   /* Learning rate for action weights, w. */
#    define BETA 0.5f    /* Learning rate for critic weights, v. */
#    define GAMMA 0.95f  /* Discount factor for critic. */
#    define LAMBDAw 0.9f /* Decay rate for w eligibility trace. */
#    define LAMBDAv 0.8f /* Decay rate for v eligibility trace. */

#    define MAX_FAILURES 1000 /* Termination criterion. */
#    define MAX_STEPS 100000

typedef float vector[N_BOXES];

int main()
{
  float x,				/* cart position, meters */
		x_dot,			/* cart velocity */
		theta,			/* pole angle, radians */
		theta_dot;		/* pole angular velocity */
  vector  w,			/* vector of action weights */
		  v,			/* vector of critic weights */
		  e,			/* vector of action weight eligibilities */
		  xbar;			/* vector of critic weight eligibilities */
  float p, oldp, rhat, r;
  int box, i, y, steps = 0, failures=0, failed;

  /*--- Initialize action and heuristic critic weights and traces. ---*/
  for (i = 0; i < N_BOXES; i++)
	w[i] = v[i] = xbar[i] = e[i] = 0.0f;

  /*--- Starting state is (0 0 0 0) ---*/
  x = x_dot = theta = theta_dot = 0.0f;

  /*--- Find box in state space containing start state ---*/
  box = get_box(x, x_dot, theta, theta_dot);

  /*--- Iterate through the action-learn loop. ---*/
  while (steps++ < MAX_STEPS && failures < MAX_FAILURES)
	{
	  /*--- Choose action randomly, biased by current weight. ---*/
	  y = (vs_uniform(0.0f, 1.0f) < prob_push_right(w[box]));

	  /*--- Update traces. ---*/
	  e[box] += (1.0f - LAMBDAw) * (y - 0.5f);
	  xbar[box] += (1.0f - LAMBDAv);

	  /*--- Remember prediction of failure for current state ---*/
	  oldp = v[box];

	  /*--- Apply action to the simulated cart-pole ---*/
	  cart_pole(y, &x, &x_dot, &theta, &theta_dot);

	  /*--- Get box of state space containing the resulting state. ---*/
	  box = get_box(x, x_dot, theta, theta_dot);

	  if (box < 0)
	{
	  /*--- Failure occurred. ---*/
	  failed = 1;
	  failures++;
	  printf("Trial %d was %d steps.\n", failures, steps);
	  steps = 0;

	  /*--- Reset state to (0 0 0 0).  Find the box. ---*/
	  x = x_dot = theta = theta_dot = 0.0f;
	  box = get_box(x, x_dot, theta, theta_dot);

	  /*--- Reinforcement upon failure is -1. Prediction of failure is 0. ---*/
	  r = -1.0f;
	  p = 0.;
	}
	  else
	{
	  /*--- Not a failure. ---*/
	  failed = 0;

	  /*--- Reinforcement is 0. Prediction of failure given by v weight. ---*/
	  r = 0;
	  p= v[box];
	}

	  /*--- Heuristic reinforcement is:	  current reinforcement
		  + gamma * new failure prediction - previous failure prediction ---*/
	  rhat = r + GAMMA * p - oldp;

	  for (i = 0; i < N_BOXES; i++)
	{
	  /*--- Update all weights. ---*/
	  w[i] += ALPHA * rhat * e[i];
	  v[i] += BETA * rhat * xbar[i];
	  if (v[i] < -1.0f)
		v[i] = v[i];

	  if (failed)
		{
		  /*--- If failure, zero all traces. ---*/
		  e[i] = 0.0f;
		  xbar[i] = 0.0f;
		}
	  else
		{
		  /*--- Otherwise, update (decay) the traces. ---*/
		  e[i] *= LAMBDAw;
		  xbar[i] *= LAMBDAv;
		}
	}

	}
  if (failures == MAX_FAILURES)
	printf("Pole not balanced. Stopping after %d failures.",failures);
  else
	printf("Pole balanced successfully for at least %d steps\n", steps);

  return 0;
}

/*----------------------------------------------------------------------
   cart_pole:  Takes an action (0 or 1) and the current values of the
 four state variables and updates their values by estimating the state
 TAU seconds later.
----------------------------------------------------------------------*/

/*** Parameters for simulation ***/

#    define GRAVITY 9.81f
#    define MASSCART 1.0f
#    define MASSPOLE 0.1f
#    define TOTAL_MASS ( MASSPOLE + MASSCART )
#    define LENGTH 0.5f /* actually half the pole's length */
#    define POLEMASS_LENGTH ( MASSPOLE * LENGTH )
#    define FORCE_MAG 10.0f
#    define TAU 0.02f /* seconds between state updates */
#    define FOURTHIRDS 1.3333333333333f


void cart_pole(int action, float *x, float *x_dot, float *theta, float *theta_dot)
{
	float xacc,thetaacc,force,costheta,sintheta,temp;

	force = (action>0)? FORCE_MAG : -FORCE_MAG;
	costheta = cos(*theta);
	sintheta = sin(*theta);

	temp = (force + POLEMASS_LENGTH * *theta_dot * *theta_dot * sintheta)
				 / TOTAL_MASS;

	thetaacc = (GRAVITY * sintheta - costheta* temp)
		   / (LENGTH * (FOURTHIRDS - MASSPOLE * costheta * costheta
											  / TOTAL_MASS));

	xacc  = temp - POLEMASS_LENGTH * thetaacc* costheta / TOTAL_MASS;

/*** Update the four state variables, using Euler's method. ***/

	*x	+= TAU * *x_dot;
	*x_dot += TAU * xacc;
	*theta += TAU * *theta_dot;
	*theta_dot += TAU * thetaacc;
}

/*----------------------------------------------------------------------
   get_box:	 Given the current state, returns a number from 1 to 162
  designating the region of the state space encompassing the current state.
  Returns a value of -1 if a failure state is encountered.
----------------------------------------------------------------------*/

#    define one_degree 0.0174532f /* 2pi/360 */
#    define six_degrees 0.1047192f
#    define twelve_degrees 0.2094384f
#    define fifty_degrees 0.87266f

int get_box(float x, float x_dot, float theta, float theta_dot)
{
  int box=0;

  if (x < -2.4f ||
	  x > 2.4f	||
	  theta < -twelve_degrees ||
	  theta > twelve_degrees)		   return(-1); /* to signal failure */

  if (x < -0.8f)						box = 0;
  else if (x < 0.8f)					box = 1;
  else									box = 2;

  if (x_dot < -0.5f);
  else if (x_dot < 0.5f)				box += 3;
  else									box += 6;

  if (theta < -six_degrees);
  else if (theta < -one_degree)			box += 9;
  else if (theta < 0.0f)				box += 18;
  else if (theta < one_degree)			box += 27;
  else if (theta < six_degrees)			box += 36;
  else									box += 45;

  if (theta_dot < -fifty_degrees);
  else if (theta_dot < fifty_degrees)	box += 54;
  else									box += 108;

  return(box);
}

#endif

#if 1

/*

Nonlinear TD/Backprop pseudo C-code

Written by Allen Bonde Jr. and Rich Sutton in April 1992.
Updated in June and August 1993.
Copyright 1993 GTE Laboratories Incorporated. All rights reserved.
Permission is granted to make copies and changes, with attribution,
for research and educational purposes.

This pseudo-code implements a fully-connected two-adaptive-layer network
learning to predict discounted cumulative outcomes through Temporal
Difference learning, as described in Sutton (1988), Barto et al. (1983),
Tesauro (1992), Anderson (1986), Lin (1992), Dayan (1992), et alia. This
is a straightforward combination of discounted TD(lambda) with
backpropagation (Rumelhart, Hinton, and Williams, 1986). This is vanilla
backprop; not even momentum is used. See Sutton & Whitehead (1993) for
an argument that backprop is not the best structural credit assignment
method to use in conjunction with TD. Discounting can be eliminated for
absorbing problems by setting GAMMA=1. Eligibility traces can be
eliminated by setting LAMBDA=0. Setting both of these parameters to 0
should give standard backprop except where the input_net at time t has its
target presented at time t+1.

This is pseudo code: before it can be run it needs I/O, a random
number generator, library links, and some declarations.	 We welcome
extensions by others converting this to immediately usable C code.

The network is structured using simple array data structures as follows:


                    OUTPUT

                  ()  ()  ()  output_net[k]
                 /	\/	\/	\	   k=0...N_OUTPUT-1
     output_trace[j][k]	  /	  output_weights[j][k]	\
               /			  \
              ()  ()  ()  ()  ()  hidden_net[j]
               \			  /		   j=0...N_HIDDEN
   hidden_trace[i][j][k]  \	  hidden_weights[i][j]	/
                 \	/\	/\	/
                  ()  ()  ()  input_net[i]
                                   i=0...N_INPUT
                     INPUT


where input_net, hidden_net, and output_net are (arrays holding) the activity
levels of the input_net, hidden, and output units respectively, hidden_weights
and output_weights are the first and second layer weights, and hidden_trace and
output_trace are the eligibility traces for the first and second layers (see
Sutton, 1989). Not explicitly shown in the figure are the biases or threshold
weights. The first layer bias is provided by a dummy nth input_net unit, and
the second layer bias is provided by a dummy (num-hidden)th hidden unit. The
activities of both of these dummy units are held at a constant value (BIAS).

In addition to the main program, this file contains 4 routines:

    init_network, which initializes the network data structures.

    feedforward, which does the forward propagation, the computation of all
        unit activities based on the current input_net and weights.

    td_backpropagate, which does the backpropagation of the TD errors, and updates
        the weights.

    eligibility_traces, which updates the eligibility traces.

These routines do all their communication through the global variables
shown in the diagram above, plus previous_output_net, an array holding a copy of the
last time step's output-layer activities.

REFERENCES

Anderson, C.W. (1986) Learning and Problem Solving with Multilayer
Connectionist Systems, UMass. PhD dissertation, dept. of Computer and
Information Science, Amherts, MA 01003.

Barto, A.G., Sutton, R.S., & Anderson, C.W. (1983) "Neuron-like adaptive
elements that can solve difficult learning control problems," IEEE
Transactions on Systems, Man, and Cybernetics SMC-13: 834-846.

Dayan, P. "The convergence of TD(lambda) for general lambda,"
Machine Learning 8: 341-362.

Lin, L.-J. (1992) "Self-improving reactive agents based on reinforcement
learning, planning and teaching," Machine Learning 8: 293-322.

Rumelhart, D.E., Hinton, G.E., & Williams, R.J. (1986) "Learning
internal representations by td_error propagation," in D.E. Rumehart & J.L.
McClelland (Eds.), Parallel Distributed Processing: Explorations in the
Microstructure of Cognition, Volume 1: Foundations, 318-362. Cambridge,
MA: MIT Press.

Sutton, R.S. (1988) "Learning to predict by the methods of temporal
differences," Machine Learning 3: 9-44.

Sutton, R.S. (1989) "Implementation details of the TD(lambda) procedure
for the case of vector predictions and backpropagation," GTE
Laboratories Technical Note TN87-509.1, May 1987, corrected August 1989.
Available via ftp from ftp.gte.com as
/pub/reinforcement-learning/sutton-TD-backprop.ps

Sutton, R.S., Whitehead, S.W. (1993) "Online learning with random
representations," Proceedings of the Tenth National Conference on
Machine Learning, 314-321. Soon to be available via ftp from ftp.gte.com
as /pub/reinforcement-learning/sutton-whitehead-93.ps.Z

Tesauro, G. (1992) "Practical issues in temporal difference learning,"
Machine Learning 8: 257-278.

*/

#    include <stdio.h>
#    include <stdlib.h>
#    include <math.h>
#    include <string.h>

typedef __int32 sword;
typedef unsigned __int32 uword;

static inline float fsqrf ( const float x ) { return ( x * x ); }

// Cart & Pole

#    define length 1.0f           // - average length of pole (half it's real length) [m]
#    define mass_of_cart 1.0f     // - mass of cart [kg]
#    define mass_of_pole 1.0f     // - mass of pole [kg]
#    define acceleration 9.80665f // - acceleration due to gravity [m/s²]
#    define time_step 0.1f        // - time step [s]
#    define steps 10              // - simulation steps in one time step

// Experimental Parameters:

#    define N_INPUT 4  // Number of inputs
#    define N_HIDDEN 5 // Number of hidden units
#    define N_OUTPUT 1 // Number of output units

#    define ALPHA ( 1.0f / ( N_INPUT ) ) // 1st layer learning rate (typically 1/N_INPUT)
#    define BETA ( 1.0f / ( N_HIDDEN ) ) // 2nd layer learning rate (typically 1/N_HIDDEN)
#    define GAMMA 0.9f                   // Discount-rate parameter (typically 0.9)

#    define LAMBDA 0.1f     // Trace decay parameter (should be <= gamma)
#    define TIME_STEPS 1024 // Number of time steps to simulate
#    define N_EPOCHS 102400

typedef struct pole {

    float position;         // - position of cart [m]
    float velocity;         // - velocity of cart [m/s]
    float angle;            // - angle of pole [°]
    float angular_velocity; // - angular velocity of pole [°/s]

    float force; // - force applied to cart

} pole; //	  during current time step [N]

pole p[ TIME_STEPS ] = { 0 };

typedef struct nds {

    // Network Data Structure:

    float hidden_weights[ N_INPUT + 1 ][ N_HIDDEN ];  // Hidden layer weights
    float output_weights[ N_HIDDEN + 1 ][ N_OUTPUT ]; // Output layer weights

    float input_net[ TIME_STEPS ][ N_INPUT + 1 ]; // Input data (units+bias)

    float hidden_net[ N_HIDDEN + 1 ]; // Hidden layer activation
    float output_net[ N_OUTPUT ];     // Output layer activation

} nds;

typedef struct lds {

    // Learning Data Structure:

    float previous_output_net[ N_OUTPUT ];

    float hidden_trace[ N_INPUT + 1 ][ N_HIDDEN + 1 ][ N_OUTPUT ]; // Hidden layer trace
    float output_trace[ N_HIDDEN + 1 ][ N_OUTPUT ];                // Output layer trace

    float reward[ TIME_STEPS ][ N_OUTPUT ]; // Reward

} lds;

nds nn = { 0 };
lds td = { 0 };

void td_initiate ( void ) {

    // Initialize weights and biases

    for ( uword t = 0; t < TIME_STEPS; t++ )
        // Set biases
        nn.input_net[ t ][ N_INPUT ] = 1.0f;

    nn.hidden_net[ N_HIDDEN ] = 1.0f;

    vs_rng_uniform ( ( N_INPUT + 1 ) * N_HIDDEN, ( float * ) nn.hidden_weights, -0.1f, 0.1f );
    vs_rng_uniform ( ( N_HIDDEN + 1 ) * N_OUTPUT, ( float * ) nn.output_weights, -0.1f, 0.1f );
}

static inline void _feedforward ( const float * input_net ) { // Bias is implicit!

    // Compute hidden layer and output predictions (feed_forward)

    for ( uword j = 0; j < N_HIDDEN; j++ ) {

        nn.hidden_net[ j ] = 0.0f;

        uword i = 0;

        for ( ; i < N_INPUT; i++ )

            nn.hidden_net[ j ] += input_net[ i ] * nn.hidden_weights[ i ][ j ];

        nn.hidden_net[ j ] = 1.0f / ( 1.0f + expf ( -nn.hidden_net[ j ] + nn.hidden_weights[ i ][ j ] ) ); // Asymmetric sigmoid
    }

    for ( uword k = 0; k < N_OUTPUT; k++ ) {

        nn.output_net[ k ] = 0.0f;

        for ( uword j = 0; j <= N_HIDDEN; j++ )

            nn.output_net[ k ] += nn.hidden_net[ j ] * nn.output_weights[ j ][ k ];

        nn.output_net[ k ] = 1.0f / ( 1.0f + expf ( -nn.output_net[ k ] ) ); // Asymmetric sigmoid (optional)
    }
}

static inline void feedforward ( const uword t ) {

    // Compute hidden layer and output predictions (feed_forward)

    for ( uword j = 0; j < N_HIDDEN; j++ ) {

        nn.hidden_net[ j ] = 0.0f;

        for ( uword i = 0; i <= N_INPUT; i++ )

            nn.hidden_net[ j ] += nn.input_net[ t ][ i ] * nn.hidden_weights[ i ][ j ];

        nn.hidden_net[ j ] = 1.0f / ( 1.0f + expf ( -nn.hidden_net[ j ] ) ); // Asymmetric sigmoid
    }

    for ( uword k = 0; k < N_OUTPUT; k++ ) {

        nn.output_net[ k ] = 0.0f;

        for ( uword j = 0; j <= N_HIDDEN; j++ )

            nn.output_net[ k ] += nn.hidden_net[ j ] * nn.output_weights[ j ][ k ];

        nn.output_net[ k ] = 1.0f / ( 1.0f + expf ( -nn.output_net[ k ] ) ); // Asymmetric sigmoid (optional)
    }
}

static inline void _eligibility_traces ( const float * input_net ) {

    _feedforward ( input_net );

    // Save old output to previous output

    memcpy ( td.previous_output_net, nn.output_net, N_OUTPUT * sizeof ( float ) );

    // Calculate new weight eligibilities

    for ( uword k = 0; k < N_OUTPUT; k++ )

        nn.output_net[ k ] *= ( 1.0f - nn.output_net[ k ] ); // Turn output_net into derivative_output_net

    for ( uword j = 0; j <= N_HIDDEN; j++ ) {

        const float derivative_hidden_net = nn.hidden_net[ j ] * ( 1.0f - nn.hidden_net[ j ] );

        for ( uword k = 0; k < N_OUTPUT; k++ ) {

            td.output_trace[ j ][ k ] =
                LAMBDA * td.output_trace[ j ][ k ] + /* derivative_ */ nn.output_net[ k ] * nn.hidden_net[ j ];

            uword i = 0;

            for ( ; i < N_INPUT; i++ )

                td.hidden_trace[ i ][ j ][ k ] =
                    LAMBDA * td.hidden_trace[ i ][ j ][ k ] +
                    /* derivative_ */ nn.output_net[ k ] * nn.output_weights[ j ][ k ] * derivative_hidden_net * input_net[ i ];

            td.hidden_trace[ i ][ j ][ k ] = LAMBDA * td.hidden_trace[ i ][ j ][ k ] + /* derivative_ */ nn.output_net[ k ] *
                                                                                           nn.output_weights[ j ][ k ] *
                                                                                           derivative_hidden_net;
        }
    }
}

static inline void eligibility_traces ( const uword t ) {

    feedforward ( t );

    // Save old output to previous output

    memcpy ( td.previous_output_net, nn.output_net, N_OUTPUT * sizeof ( float ) );

    // Calculate new weight eligibilities

    for ( uword k = 0; k < N_OUTPUT; k++ )

        nn.output_net[ k ] *= ( 1.0f - nn.output_net[ k ] ); // Turn output_net into derivative_output_net

    for ( uword j = 0; j <= N_HIDDEN; j++ ) {

        const float derivative_hidden_net = nn.hidden_net[ j ] * ( 1.0f - nn.hidden_net[ j ] );

        for ( uword k = 0; k < N_OUTPUT; k++ ) {

            td.output_trace[ j ][ k ] =
                LAMBDA * td.output_trace[ j ][ k ] + /* derivative_ */ nn.output_net[ k ] * nn.hidden_net[ j ];

            for ( uword i = 0; i <= N_INPUT; i++ )

                td.hidden_trace[ i ][ j ][ k ] =
                    LAMBDA * td.hidden_trace[ i ][ j ][ k ] + /* derivative_ */ nn.output_net[ k ] * nn.output_weights[ j ][ k ] *
                                                                  derivative_hidden_net * nn.input_net[ t ][ i ];
        }
    }
}

void _td_learn ( const float * input_net, const float * reward ) {

    _feedforward ( input_net );

    // Update weight vectors

    for ( uword k = 0; k < N_OUTPUT; k++ ) {

        // Form errors

        const float td_error      = reward[ k ] + GAMMA * nn.output_net[ k ] - td.previous_output_net[ k ],
                    beta_td_error = BETA * td_error, alpha_td_error = ALPHA * td_error;

        for ( uword j = 0; j <= N_HIDDEN; j++ ) {

            nn.output_weights[ j ][ k ] += beta_td_error * td.output_trace[ j ][ k ];

            for ( uword i = 0; i <= N_INPUT; i++ )

                nn.hidden_weights[ i ][ j ] += alpha_td_error * td.hidden_trace[ i ][ j ][ k ];
        }
    }

    _eligibility_traces ( input_net );
}

void td_learn ( const uword t ) {

    feedforward ( t );

    // Update weight vectors

    for ( uword k = 0; k < N_OUTPUT; k++ ) {

        // Form errors

        const float td_error      = td.reward[ t ][ k ] + GAMMA * nn.output_net[ k ] - td.previous_output_net[ k ],
                    beta_td_error = BETA * td_error, alpha_td_error = ALPHA * td_error;

        for ( uword j = 0; j <= N_HIDDEN; j++ ) {

            nn.output_weights[ j ][ k ] += beta_td_error * td.output_trace[ j ][ k ];

            for ( uword i = 0; i <= N_INPUT; i++ )

                nn.hidden_weights[ i ][ j ] += alpha_td_error * td.hidden_trace[ i ][ j ][ k ];
        }
    }

    eligibility_traces ( t );
}

void pole_initiate ( ) {

    p[ 0 ].position = 0.0f;
    p[ 0 ].velocity = 0.0f;

    do
        p[ 0 ].angle = vs_uniform ( -3.0f, 3.0f );

    while ( p[ 0 ].angle == 0.0f );

    p[ 0 ].angular_velocity = 0.0f;
    p[ 0 ].force            = 0.0f;
}

void simulate_time_step_pole ( const uword t ) { // radians !!

    float position = p[ t - 1 ].position, velocity = p[ t - 1 ].velocity, angle = p[ t - 1 ].angle /* * 0.01745329238474369f */,
          angular_velocity = p[ t - 1 ].angular_velocity /* * 0.01745329238474369f */;
    const float force      = p[ t - 1 ].force;

    for ( uword s = 0; s < steps; s++ ) {

        const float derivative_angular_velocity =
            ( acceleration * sinf ( angle ) +
              cosf ( angle ) * ( ( -force - mass_of_pole * length * fsqrf ( angular_velocity ) * sinf ( angle ) ) /
                                 ( mass_of_cart + mass_of_pole ) ) ) /
            ( length * ( 4.0f / 3.0f - ( mass_of_pole * fsqrf ( cosf ( angle ) ) ) / ( mass_of_cart + mass_of_pole ) ) );

        position += ( time_step / steps ) * velocity;
        velocity += ( time_step / steps ) *
                    ( force + mass_of_pole * length *
                                  ( fsqrf ( angular_velocity ) * sinf ( angle ) - derivative_angular_velocity * cosf ( angle ) ) ) /
                    ( mass_of_cart + mass_of_pole );
        angle += ( time_step / steps ) * angular_velocity;
        angular_velocity += ( time_step / steps ) * derivative_angular_velocity;
    }

    p[ t ].position         = position;
    p[ t ].velocity         = velocity;
    p[ t ].angle            = angle /* * 57.295780181884766f */;
    p[ t ].angular_velocity = angular_velocity /* * 57.295780181884766f */;
}

bool is_pole_balanced ( const uword t ) {

    return ( p[ t ].angle >= ( -60.0f * 0.01745329238474369f ) ) && ( p[ t ].angle <= ( 60.0f * 0.01745329238474369f ) );
}

float score_of_pole ( const uword t ) {

    return -fsqrf ( p[ t ].angle / 0.01745329238474369f ); // ?
}

static inline void copy_pole_to_nds ( const uword t ) {

    float * i      = &nn.input_net[ t ][ 0 ];
    const pole * j = p + t;

    *i++ = j->position;
    *i++ = j->velocity;
    *i++ = j->angle;
    *i   = j->angular_velocity;
}

static inline void copy_lds_to_pole ( const uword t ) {

    p[ t ].force = nn.output_net[ 0 ];

    printf ( "%f\n", score_of_pole ( t ) );
}

void td_epoch ( ) {

    uword t = 0;

    eligibility_traces ( t );

    for ( t = 1; t < TIME_STEPS; t++ ) {

        simulate_time_step_pole ( t );
        copy_pole_to_nds ( t );

        if ( !is_pole_balanced ( t ) ) {

            td.reward[ t ][ 0 ] = -1.0f;

            td_learn ( t );
            copy_lds_to_pole ( t );

            return;
        }

        td_learn ( t );
        copy_lds_to_pole ( t );
    }

    td.reward[ t ][ 0 ] = 1.0f;

    td_learn ( t );
    copy_lds_to_pole ( t );
}

sword main ( void ) {

    td_initiate ( );

    for ( uword e = 0; e < N_EPOCHS; e++ ) {

        pole_initiate ( );
        td_epoch ( );
    }

    return 0;
}

#endif

#if 0

/* A simple kalman filter example by Adrian Boeing

	www.adrianboeing.com

*/

#    include <stdio.h>
#    include <stdlib.h>
#    include <imath.h>

double frand() {

	return 2*((rand()/(double)RAND_MAX) - 0.5);
}

int main ( ) {

	//initial values for the kalman filter
	float x_est_last = 0;
	float P_last = 0;
	//the noise in the system
	float Q = 0.022;
	float R = 0.617;

	float K;
	float P;
	float P_temp;
	float x_temp_est;
	float x_est;
	float z_measured; //the 'noisy' value we measured
	float z_real = 0.5; //the ideal value we wish to measure

	srand(0);

	//initialize with a measurement
	x_est_last = z_real + frand()*0.09;

	float sum_error_kalman = 0;
	float sum_error_measure = 0;

	for (int i=0;i<30;i++) {
		//do a prediction
		x_temp_est = x_est_last;
		P_temp = P_last + Q;
		//calculate the Kalman gain
		K = P_temp * (1.0/(P_temp + R));
		//measure
		z_measured = z_real + frand()*0.09; //the real measurement plus noise
		//correct
		x_est = x_temp_est + K * (z_measured - x_temp_est);
		P = (1- K) * P_temp;
		//we have our new system

		printf("Ideal	 position: %6.3f \n",z_real);
		printf("Mesaured position: %6.3f [diff:%.3f]\n",z_measured,fabs(z_real-z_measured));
		printf("Kalman	 position: %6.3f [diff:%.3f]\n",x_est,fabs(z_real - x_est));

		sum_error_kalman += fabs(z_real - x_est);
		sum_error_measure += fabs(z_real-z_measured);

		//update our last's
		P_last = P;
		x_est_last = x_est;
	}

	printf("Total error if using raw measured:	%f\n",sum_error_measure);
	printf("Total error if using kalman filter: %f\n",sum_error_kalman);
	printf("Reduction in error: %d%% \n",100-(int)((sum_error_kalman/sum_error_measure)*100));


	return 0;
}
#endif

/************************************************************************

http://www.incompleteideas.net/

http://www.incompleteideas.net/td-backprop-pseudo-code.text

Nonlinear TD/Backprop pseudo C-code

Written by Allen Bonde Jr. and Rich Sutton in April 1992.
Updated in June and August 1993.
Copyright 1993 GTE Laboratories Incorporated. All rights reserved.
Permission is granted to make copies and changes, with attribution,
for research and educational purposes.

This pseudo-code implements a fully-connected two-adaptive-layer network
learning to predict discounted cumulative outcomes through Temporal
Difference learning, as described in Sutton (1988), Barto et al. (1983),
Tesauro (1992), Anderson (1986), Lin (1992), Dayan (1992), et alia. This
is a straightforward combination of discounted TD(lambda) with
backpropagation (Rumelhart, Hinton, and Williams, 1986). This is vanilla
backprop; not even momentum is used. See Sutton & Whitehead (1993) for
an argument that backprop is not the best structural credit assignment
method to use in conjunction with TD. Discounting can be eliminated for
absorbing problems by setting GAMMA=1. Eligibility traces can be
eliminated by setting LAMBDA=0. Setting both of these parameters to 0
should give standard backprop except where the input at time t has its
target presented at time t+1.

This is pseudo code: before it can be run it needs I/O, a random
number generator, library links, and some declarations.  We welcome
extensions by others converting this to immediately usable C code.

The network is structured using simple array data structures as follows:


                    OUTPUT

                  ()  ()  ()  y[k]
                 /  \/  \/  \      k=0...m-1
     ew[j][k]   /   w[j][k]  \
               /              \
              ()  ()  ()  ()  ()  h[j]
               \              /        j=0...num_hidden
   ev[i][j][k]  \   v[i][j]  /
                 \  /\  /\  /
                  ()  ()  ()  x[i]
                                   i=0...n
                     INPUT


where x, h, and y are (arrays holding) the activity levels of the input,
hidden, and output units respectively, v and w are the first and second
layer weights, and ev and ew are the eligibility traces for the first
and second layers (see Sutton, 1989). Not explicitly shown in the figure
are the biases or threshold weights. The first layer bias is provided by
a dummy nth input unit, and the second layer bias is provided by a dummy
(num-hidden)th hidden unit. The activities of both of these dummy units
are held at a constant value (BIAS).

In addition to the main program, this file contains 4 routines:

    init_network, which initializes the network data structures.

    response, which does the forward propagation, the computation of all
        unit activities based on the current input and weights.

    td_learn, which does the backpropagation of the TD errors, and updates
        the weights.

    update_eligibilities, which updates the eligibility traces.

These routines do all their communication through the global variables
shown in the diagram above, plus old_y, an array holding a copy of the
last time step's output-layer activities.

For simplicity, all the array dimensions are specified as MAX_UNITS, the
maximum allowed number of units in any layer.  This could of course be
tightened up if memory becomes a problem.

REFERENCES

Anderson, C.W. (1986) Learning and Problem Solving with Multilayer
Connectionist Systems, UMass. PhD dissertation, dept. of Computer and
Information Science, Amherts, MA 01003.

Barto, A.G., Sutton, R.S., & Anderson, C.W. (1983) "Neuron-like adaptive
elements that can solve difficult learning control problems," IEEE
Transactions on Systems, Man, and Cybernetics SMC-13: 834-846.

Dayan, P. "The convergence of TD(lambda) for general lambda,"
Machine Learning 8: 341-362.

Lin, L.-J. (1992) "Self-improving reactive agents based on reinforcement
learning, planning and teaching," Machine Learning 8: 293-322.

Rumelhart, D.E., Hinton, G.E., & Williams, R.J. (1986) "Learning
internal representations by error propagation," in D.E. Rumehart & J.L.
McClelland (Eds.), Parallel Distributed Processing: Explorations in the
Microstructure of Cognition, Volume 1: Foundations, 318-362. Cambridge,
MA: MIT Press.

Sutton, R.S. (1988) "Learning to predict by the methods of temporal
differences," Machine Learning 3: 9-44.

Sutton, R.S. (1989) "Implementation details of the TD(lambda) procedure
for the case of vector predictions and backpropagation," GTE
Laboratories Technical Note TN87-509.1, May 1987, corrected August 1989.
Available via ftp from ftp.gte.com as
/pub/reinforcement-learning/sutton-TD-backprop.ps

Sutton, R.S., Whitehead, S.W. (1993) "Online learning with random
representations," Proceedings of the Tenth National Conference on
Machine Learning, 314-321. Soon to be available via ftp from ftp.gte.com
as /pub/reinforcement-learning/sutton-whitehead-93.ps.Z

Tesauro, G. (1992) "Practical issues in temporal difference learning,"
Machine Learning 8: 257-278.
************************************************************************/

// Experimental Parameters:

int n, num_hidden, m; // number of inputs, hidden, and output units
int MAX_UNITS;        // maximum total number of units (to set array sizes)
int time_steps;       // number of time steps to simulate

float BIAS;   // strength of the bias (constant input) contribution
float ALPHA;  // 1st layer learning rate (typically 1/n)
float BETA;   // 2nd layer learning rate (typically 1/num_hidden)
float GAMMA;  // discount-rate parameter (typically 0.9)
float LAMBDA; // trace decay parameter (should be <= gamma)

// Network Data Structure:

float x[ time_steps ][ MAX_UNITS ]; // input data (units)
float h[ MAX_UNITS ];               // hidden layer
float y[ MAX_UNITS ];               // output layer
float v[ MAX_UNITS ];               // ? layer
float w[ MAX_UNITS ][ MAX_UNITS ];  // weights */

// Learning Data Structure: */

float old_y[ MAX_UNITS ];
float ev[ MAX_UNITS ][ MAX_UNITS ][ MAX_UNITS ]; // hidden trace
float ew[ MAX_UNITS ][ MAX_UNITS ];              // output trace
float r[ time_steps ][ MAX_UNITS ];              // reward
float error[ MAX_UNITS ];                        // TD error
int t;                                           // current time step

int main ( ) {

    int k;

    init_network ( );

    t = 0;        // no learning on time step 0
    response ( ); // just compute old response (old_y)

    for ( k = 0; k < m; ++k ) //
        old_y[ k ] = y[ k ];  //

    update_eligibilities ( ); //...and prepare the eligibilities

    for ( t = 1; t <= time_steps; ++t ) {                           // a single pass through time series data
        response ( );                                               // forward pass - compute activities
        for ( k = 0; k < m; ++k )                                   //
            error[ k ] = r[ t ][ k ] + GAMMA * y[ k ] - old_y[ k ]; // form errors
        td_learn ( );                                               // backward pass - learning
        response ( );                                               // forward pass must be done twice to form TD errors
        for ( k = 0; k < m; ++k )                                   //
            old_y[ k ] = y[ k ];                                    // for use in next cycle's TD errors
        update_eligibilities ( );                                   // update eligibility traces
    }                                                               // end t

    return EXIT_SUCCESS;
}

// Initialize weights and biases.
void init_network ( ) noexcept {

    int i = 0;

    for ( int s = 0; s < time_steps; ++s )
        x[ s ][ n ] = BIAS;

    h[ num_hidden ] = BIAS;

    for ( int j = 0; j <= num_hidden; ++j ) {
        for ( int k = 0; k < m; ++k ) {
            w[ j ][ k ]  = some_small_random_value;
            ew[ i ][ k ] = { };
            old_y[ k ]   = { };
        }
        for ( i = 0; i <= n; ++i ) {
            v[ i ][ j ] = some_small_random_value;
            for ( k = 0; k < m; ++k )
                ev[ i ][ j ][ k ] = { };
        }
    }
}

// Compute hidden layer and output predictions.
void response ( ) noexcept {

    h[ num_hidden ] = BIAS;
    x[ t ][ n ]     = BIAS;

    for ( int j = 0; j < num_hidden; ++j ) {
        h[ j ] = { };
        for ( int i = 0; i <= n; ++i )
            h[ j ] += x[ t ][ i ] * v[ i ][ j ];
        h[ j ] = 1.0f / ( 1.0f + std::exp ( -h[ j ] ) ); // asymmetric sigmoid
    }

    for ( int k = 0; k < m; ++k ) {
        y[ k ] = { };
        for ( int j = 0; j <= num_hidden; ++j )
            y[ k ] += h[ j ] * w[ j ][ k ];
        y[ k ] = 1.0f / ( 1.0f + std::exp ( -y[ k ] ) ); // asymmetric sigmoid (OPTIONAL)
    }
}

// Update weight vectors.
void td_learn ( ) noexcept {

    for ( int k = 0; k < m; ++k ) {
        for ( int j = 0; j <= num_hidden; ++j ) {
            w[ j ][ k ] += BETA * error[ k ] * ew[ j ][ k ];
            for ( int i = 0; i <= n; ++i )
                v[ i ][ j ] += ALPHA * error[ k ] * ev[ i ][ j ][ k ];
        }
    }
}

// Calculate new weight eligibilities.
void update_eligibilities ( ) noexcept {

    float temp[ MAX_UNITS ];

    for ( int k = 0; k < m; ++k )
        temp[ k ] = y[ k ] * ( 1.0f - y[ k ] );

    for ( int j = 0; j <= num_hidden; ++j ) {
        for ( int k = 0; k < m; ++k ) {
            ew[ j ][ k ] = LAMBDA * ew[ j ][ k ] + temp[ k ] * h[ j ];
            for ( int i = 0; i <= n; ++i )
                ev[ i ][ j ][ k ] = LAMBDA * ev[ i ][ j ][ k ] + temp[ k ] * w[ j ][ k ] * h[ j ] * ( 1.0f - h[ j ] ) * x[ t ][ i ];
        }
    }
}
