!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! TYPE: MODULE
! CODE: FORTRAN 90
! This module contains all the subroutines used to calculate
! the first-arrival traveltime field through the grid.
! Subroutines are:
! (1) travel
! (2) fouds1
! (3) fouds2
! (4) addtree
! (5) downtree
! (6) updtree
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
MODULE traveltime
USE globalp
IMPLICIT NONE
INTEGER ntr
TYPE backpointer
   INTEGER :: px,pz
END TYPE backpointer
TYPE(backpointer), DIMENSION (:), ALLOCATABLE :: btg
!$omp threadprivate (ntr,btg)
!
! btg = backpointer to relate grid nodes to binary tree entries
! px = grid-point in x
! pz = grid-point in z
! ntr = number of entries in binary tree
!

CONTAINS

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! TYPE: SUBROUTINE
! CODE: FORTRAN 90
! This subroutine is passed the location of a source, and from
! this point the first-arrival traveltime field through the
! velocity grid is determined.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
SUBROUTINE travel(scx,scz,urg)
IMPLICIT NONE
INTEGER :: isx,isz,sw,i,j,ix,iz,urg,swrg
REAL(KIND=i10) :: scx,scz,vsrc,dsx,dsz,ds
REAL(KIND=i10), DIMENSION (2,2) :: vss
! isx,isz = grid cell indices (i,j,k) which contains source
! scx,scz = (r,x,y) location of source
! sw = a switch (0=off,1=on)
! ix,iz = j,k position of "close" point with minimum traveltime
! maxbt = maximum size of narrow band binary tree
! rd2,rd3 = substitution variables
! vsrc = velocity at source
! vss = velocity at nodes surrounding source
! dsx, dsz = distance from source to cell boundary in x and z
! ds = distance from source to nearby node
! urg = use refined grid (0=no,1=yes,2=previously used)
! swrg = switch to end refined source grid computation
!
! The first step is to find out where the source resides
! in the grid of nodes. The cell in which it resides is
! identified by the "north-west" node of the cell. If the
! source lies on the edge or corner (a node) of the cell, then
! this scheme still applies.
!
isx=floor((scx-gox)/dnx)+1
isz=floor((scz-goz)/dnz)+1
sw=0
IF(isx.lt.1.or.isx.gt.nnx)sw=1
IF(isz.lt.1.or.isz.gt.nnz)sw=1
IF(sw.eq.1)then
   WRITE(*,*)"Source lies outside bounds of model (lat,long)= ",scx, scz
   WRITE(*,*)"TERMINATING PROGRAM!!!"
   STOP
ENDIF
IF(isx.eq.nnx)isx=isx-1
IF(isz.eq.nnz)isz=isz-1
!
! Set all values of nsts to -1 if beginning from a source
! point.
!
IF(urg.NE.2)nsts=-1
!
! set initial size of binary tree to zero
!
ntr=0
IF(urg.EQ.2)THEN
!
!  In this case, source grid refinement has been applied, so
!  the initial narrow band will come from resampling the
!  refined grid.
!
   DO i=1,nnx
      DO j=1,nnz
         IF(nsts(j,i).GT.0)THEN
            CALL addtree(j,i)
         ENDIF
      ENDDO
   ENDDO
ELSE 
!
!  In general, the source point need not lie on a grid point.
!  Bi-linear interpolation is used to find velocity at the
!  source point.
!
   nsts=-1
   DO i=1,2
      DO j=1,2
         vss(i,j)=veln(isz-1+j,isx-1+i)
      ENDDO
   ENDDO
   dsx=(scx-gox)-(isx-1)*dnx
   dsz=(scz-goz)-(isz-1)*dnz
   CALL bilinear(vss,dsx,dsz,vsrc)
!
!  Now find the traveltime at the four surrounding grid points. This
!  is calculated approximately by assuming the traveltime from the
!  source point to each node is equal to the the distance between
!  the two points divided by the average velocity of the points
!
   DO i=1,2
      DO j=1,2
         ds=SQRT((dsx-(i-1)*dnx)**2+(dsz-(j-1)*dnz)**2)
         if(earth>0) ds=SQRT(((dsx-(i-1)*dnx)*earth)**2+((dsz-(j-1)*dnz)*earth*sin(scx))**2)
         ttn(isz-1+j,isx-1+i)=2.0*ds/(vss(i,j)+vsrc)
         CALL addtree(isz-1+j,isx-1+i)
      ENDDO
   ENDDO
ENDIF
!
! Now calculate the first-arrival traveltimes at the
! remaining grid points. This is done via a loop which
! repeats the procedure of finding the first-arrival
! of all "close" points, adding it to the set of "alive"
! points and updating the points surrounding the new "alive"
! point. The process ceases when the binary tree is empty,
! in which case all grid points are "alive".
!
DO WHILE(ntr.gt.0)
!
! First, check whether source grid refinement is
! being applied; if so, then there is a special
! exit condition.
!
IF(urg.EQ.1)THEN
   ix=btg(1)%px
   iz=btg(1)%pz
   swrg=0
   IF(ix.EQ.1)THEN
      IF(vnl.NE.1)swrg=1
   ENDIF
   IF(ix.EQ.nnx)THEN
      IF(vnr.NE.nnx)swrg=1
   ENDIF
   IF(iz.EQ.1)THEN
      IF(vnt.NE.1)swrg=1
   ENDIF
   IF(iz.EQ.nnz)THEN
      IF(vnb.NE.nnz)swrg=1
   ENDIF
   IF(swrg.EQ.1)THEN
      nsts(iz,ix)=0
      EXIT
   ENDIF
ENDIF
!
! Set the "close" point with minimum traveltime
! to "alive"
!
   ix=btg(1)%px
   iz=btg(1)%pz
   nsts(iz,ix)=0
!
! Update the binary tree by removing the root and
! sweeping down the tree.
!
   CALL downtree
!
! Now update or find values of up to four grid points
! that surround the new "alive" point.
!
! Test points that vary in x
!
   DO i=ix-1,ix+1,2
      IF(i.ge.1.and.i.le.nnx)THEN
         IF(nsts(iz,i).eq.-1)THEN
!
! This option occurs when a far point is added to the list
! of "close" points
!
            IF(fom.eq.0)THEN
               CALL fouds1(iz,i)
            ELSE
               CALL fouds2(iz,i)
            ENDIF
            CALL addtree(iz,i)
         ELSE IF(nsts(iz,i).gt.0)THEN
!
! This happens when a "close" point is updated
!
            IF(fom.eq.0)THEN
               CALL fouds1(iz,i)
            ELSE
               CALL fouds2(iz,i)
            ENDIF
            CALL updtree(iz,i)
         ENDIF
      ENDIF
   ENDDO
!
! Test points that vary in z
!
   DO i=iz-1,iz+1,2
      IF(i.ge.1.and.i.le.nnz)THEN
         IF(nsts(i,ix).eq.-1)THEN
!
! This option occurs when a far point is added to the list
! of "close" points
!
            IF(fom.eq.0)THEN
               CALL fouds1(i,ix)
            ELSE
               CALL fouds2(i,ix)
            ENDIF
            CALL addtree(i,ix)
         ELSE IF(nsts(i,ix).gt.0)THEN
!
! This happens when a "close" point is updated
!
            IF(fom.eq.0)THEN
               CALL fouds1(i,ix)
            ELSE
               CALL fouds2(i,ix)
            ENDIF
            CALL updtree(i,ix)
         ENDIF
      ENDIF
   ENDDO
ENDDO
END SUBROUTINE travel

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! TYPE: SUBROUTINE
! CODE: FORTRAN 90
! This subroutine calculates a trial first-arrival traveltime
! at a given node from surrounding nodes using the
! First-Order Upwind Difference Scheme (FOUDS) of
! Sethian and Popovici (1999).
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
SUBROUTINE fouds1(iz,ix)
IMPLICIT NONE
INTEGER :: j,k,ix,iz,tsw1,swsol
REAL(KIND=i10) :: trav,travm,slown,tdsh,tref
REAL(KIND=i10) :: a,b,c,u,v,em,ri,risti
REAL(KIND=i10) :: rd1
!
! ix = NS position of node coordinate for determination
! iz = EW vertical position of node coordinate for determination
! trav = traveltime calculated for trial node
! travm = minimum traveltime calculated for trial node
! slown = slowness at (iz,ix)
! tsw1 = traveltime switch (0=first time,1=previously)
! a,b,c,u,v,em = Convenience variables for solving quadratic
! tdsh = local traveltime from neighbouring node
! tref = reference traveltime at neighbouring node
! ri = Radial distance
! risti = ri*sin(theta) at point (iz,ix)
! rd1 = dummy variable
! swsol = switch for solution (0=no solution, 1=solution)
!
! Inspect each of the four quadrants for the minimum time
! solution.
!
travm = huge(travm)
tsw1=0
slown=1.0/veln(iz,ix)
ri=earth
risti=ri*sin(gox+(ix-1)*dnx)
if(abs(earth)<1e-6)then
   ri = 1
   risti = 1
endif
DO j=ix-1,ix+1,2
   DO k=iz-1,iz+1,2 
      IF(j.GE.1.AND.j.LE.nnx)THEN
         IF(k.GE.1.AND.k.LE.nnz)THEN
!
!           There are seven solution options in
!           each quadrant.
!
            swsol=0
            IF(nsts(iz,j).EQ.0)THEN
               swsol=1
               IF(nsts(k,ix).EQ.0)THEN
                  u=ri*dnx
                  v=risti*dnz
                  em=ttn(k,ix)-ttn(iz,j)
                  a=u**2+v**2
                  b=-2.0*u**2*em
                  c=u**2*(em**2-v**2*slown**2)
                  tref=ttn(iz,j)
               ELSE
                  a=1.0
                  b=0.0
                  c=-slown**2*ri**2*dnx**2
                  !c=-slown**2*dnx**2
                  tref=ttn(iz,j)
               ENDIF
            ELSE IF(nsts(k,ix).EQ.0)THEN
               swsol=1
               a=1.0
               b=0.0
               c=-(slown*risti*dnz)**2
               !c=-(slown*dnz)**2
               tref=ttn(k,ix)
            ENDIF
!
!           Now find the solution of the quadratic equation
!
            IF(swsol.EQ.1)THEN
               rd1=b**2-4.0*a*c
               IF(rd1 >= 0.0)then
                  tdsh=(-b+sqrt(rd1))/(2.0*a)
                  trav=tref+tdsh
               endif
               ! if no root for quandratic equation or the travel time is not
               ! valid
               if(rd1<0.0 .or. trav<min(ttn(iz,j),ttn(k,ix)))then
                   if(nsts(iz,j) == 0) trav = ttn(iz,j) + slown*dnx*ri
                   if(nsts(k,ix) == 0 .and. trav > ttn(k,ix) + slown*risti*dnz)then 
                       trav = ttn(k,ix) + slown*dnz*risti
                   endif
               endif
               IF(tsw1.EQ.1)THEN
                  travm=MIN(trav,travm)
               ELSE
                  travm=trav
                  tsw1=1
               ENDIF
            ENDIF
         ENDIF
      ENDIF
   ENDDO
ENDDO
ttn(iz,ix)=travm
if(travm /= travm)then
    write(*,*) 'NaN occured'
    write(*,*) 'rd1, b, a, c: ', rd1, b, a, c
    write(*,*) 'iz, ix, time: ', iz, ix, ttn(iz,j), ttn(k,ix)
endif
END SUBROUTINE fouds1

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! TYPE: SUBROUTINE
! CODE: FORTRAN 90
! This subroutine calculates a trial first-arrival traveltime
! at a given node from surrounding nodes using the
! Mixed-Order (2nd) Upwind Difference Scheme (FOUDS) of
! Popovici and Sethian (2002).
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
SUBROUTINE fouds2(iz,ix)
IMPLICIT NONE
INTEGER :: j,k,j2,k2,ix,iz,tsw1
INTEGER :: swj,swk,swsol
REAL(KIND=i10) :: trav,travm,slown,tdsh,tref,tdiv
REAL(KIND=i10) :: a,b,c,u,v,em,ri,risti,rd1
!
! ix = NS position of node coordinate for determination
! iz = EW vertical position of node coordinate for determination
! trav = traveltime calculated for trial node
! travm = minimum traveltime calculated for trial node
! slown = slowness at (iz,ix)
! tsw1 = traveltime switch (0=first time,1=previously)
! a,b,c,u,v,em = Convenience variables for solving quadratic
! tdsh = local traveltime from neighbouring node
! tref = reference traveltime at neighbouring node
! ri = Radial distance
! risti = ri*sin(theta) at point (iz,ix)
! swj,swk = switches for second order operators
! tdiv = term to divide tref by depending on operator order
! swsol = switch for solution (0=no solution, 1=solution)
!
! Inspect each of the four quadrants for the minimum time
! solution.
!
travm = huge(travm)
tsw1=0
slown=1.0/veln(iz,ix)
ri=earth
risti=ri*sin(gox+(ix-1)*dnx)
if(abs(earth)<1e-6)then
   ri = 1
   risti = 1
endif
DO j=ix-1,ix+1,2
   IF(j.GE.1.AND.j.LE.nnx)THEN
      swj=-1
      IF(j.eq.ix-1)THEN
         j2=j-1
         IF(j2.GE.1)THEN
            IF(nsts(iz,j2).EQ.0)swj=0
         ENDIF
      ELSE
         j2=j+1
         IF(j2.LE.nnx)THEN
            IF(nsts(iz,j2).EQ.0)swj=0
         ENDIF
      ENDIF
      IF(nsts(iz,j).EQ.0.AND.swj.EQ.0)THEN
         swj=-1
         IF(ttn(iz,j).GT.ttn(iz,j2))THEN
            swj=0
         ENDIF
      ELSE
         swj=-1
      ENDIF
      DO k=iz-1,iz+1,2
         IF(k.GE.1.AND.k.LE.nnz)THEN
            swk=-1
            IF(k.eq.iz-1)THEN
               k2=k-1
               IF(k2.GE.1)THEN
                  IF(nsts(k2,ix).EQ.0)swk=0
               ENDIF
            ELSE
               k2=k+1
               IF(k2.LE.nnz)THEN
                  IF(nsts(k2,ix).EQ.0)swk=0
               ENDIF
            ENDIF
            IF(nsts(k,ix).EQ.0.AND.swk.EQ.0)THEN
               swk=-1
               IF(ttn(k,ix).GT.ttn(k2,ix))THEN
                  swk=0
               ENDIF
            ELSE
               swk=-1
            ENDIF
!
!           There are 8 solution options in
!           each quadrant.
!
            swsol=0
            IF(swj.EQ.0)THEN
               swsol=1
               IF(swk.EQ.0)THEN
                  u=2.0*ri*dnx
                  v=2.0*risti*dnz
                  !u = 2.0*dnx
                  !v = 2.0*dnz
                  em=4.0*ttn(iz,j)-ttn(iz,j2)-4.0*ttn(k,ix)
                  em=em+ttn(k2,ix)
                  a=v**2+u**2
                  b=2.0*em*u**2
                  c=u**2*(em**2-slown**2*v**2)
                  tref=4.0*ttn(iz,j)-ttn(iz,j2)
                  tdiv=3.0
               ELSE IF(nsts(k,ix).EQ.0)THEN
                  u=risti*dnz
                  v=2.0*ri*dnx
                  !u=dnz
                  !v=2.0*dnx
                  em=3.0*ttn(k,ix)-4.0*ttn(iz,j)+ttn(iz,j2)
                  a=v**2+9.0*u**2
                  b=6.0*em*u**2
                  c=u**2*(em**2-slown**2*v**2)
                  tref=ttn(k,ix)
                  tdiv=1.0
               ELSE
                  u=2.0*ri*dnx
                  !u=2.0*dnx
                  a=1.0
                  b=0.0
                  c=-u**2*slown**2
                  tref=4.0*ttn(iz,j)-ttn(iz,j2)
                  tdiv=3.0
               ENDIF
            ELSE IF(nsts(iz,j).EQ.0)THEN
               swsol=1
               IF(swk.EQ.0)THEN
                  !u=dnx
                  !v=2.0*dnz
                  u=ri*dnx
                  v=2.0*risti*dnz
                  em=3.0*ttn(iz,j)-4.0*ttn(k,ix)+ttn(k2,ix)
                  a=v**2+9.0*u**2
                  b=6.0*em*u**2
                  c=u**2*(em**2-v**2*slown**2)
                  tref=ttn(iz,j)
                  tdiv=1.0
               ELSE IF(nsts(k,ix).EQ.0)THEN
                  !u=dnx
                  !v=dnz
                  u=ri*dnx
                  v=risti*dnz
                  em=ttn(k,ix)-ttn(iz,j)
                  a=u**2+v**2
                  b=-2.0*u**2*em
                  c=u**2*(em**2-v**2*slown**2)
                  tref=ttn(iz,j)
                  tdiv=1.0
               ELSE
                  a=1.0
                  b=0.0
                  !c=-slown**2*dnx**2
                  c=-slown**2*ri**2*dnx**2
                  tref=ttn(iz,j)
                  tdiv=1.0
               ENDIF
            ELSE
               IF(swk.EQ.0)THEN
                  swsol=1
                  u=2.0*risti*dnz
                  !u=2.0*dnz
                  a=1.0
                  b=0.0
                  c=-u**2*slown**2
                  tref=4.0*ttn(k,ix)-ttn(k2,ix)
                  tdiv=3.0
               ELSE IF(nsts(k,ix).EQ.0)THEN
                  swsol=1
                  a=1.0
                  b=0.0
                  !c=-slown**2*dnz**2
                  c=-slown**2*risti**2*dnz**2
                  tref=ttn(k,ix)
                  tdiv=1.0
               ENDIF
            ENDIF
!
!           Now find the solution of the quadratic equation
!
            IF(swsol.EQ.1)THEN
               rd1=b**2-4.0*a*c
               IF(rd1>=0.0)then
                  tdsh=(-b+sqrt(rd1))/(2.0*a)
                  trav=(tref+tdsh)/tdiv
               endif
               ! if no root for quandratic equation or the travel time is not
               ! valid
               if(rd1<0.0 .or. trav<min(ttn(iz,j),ttn(k,ix)))then
                   if(nsts(iz,j) == 0) trav = ttn(iz,j) + slown*ri*dnx
                   if(nsts(k,ix) == 0 .and. trav > ttn(k,ix) + slown*risti*dnz)then 
                       trav = ttn(k,ix) + slown*dnz*risti
                   endif
               endif
               IF(tsw1.EQ.1)THEN
                  travm=MIN(trav,travm)
               ELSE
                  travm=trav
                  tsw1=1
               ENDIF
            ENDIF
         ENDIF
      ENDDO
   ENDIF
ENDDO
ttn(iz,ix)=travm
if(travm /= travm)then
    write(*,*) 'NaN occured'
    write(*,*) 'rd1, b, a, c: ', rd1, b, a, c
    write(*,*) 'iz, ix, time: ', iz, ix, ttn(iz,j), ttn(k,ix)
    write(*,*) 'slow, vel: ', slown, veln(iz,ix)
    stop
endif
END SUBROUTINE fouds2

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! TYPE: SUBROUTINE
! CODE: FORTRAN 90
! This subroutine adds a value to the binary tree by
! placing a value at the bottom and pushing it up
! to its correct position.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
SUBROUTINE addtree(iz,ix)
IMPLICIT NONE
INTEGER :: ix,iz,tpp,tpc
TYPE(backpointer) :: exch
!
! ix,iz = grid position of new addition to tree
! tpp = tree position of parent
! tpc = tree position of child
! exch = dummy to exchange btg values
!
! First, increase the size of the tree by one.
!
ntr=ntr+1
!
! Put new value at base of tree
!
nsts(iz,ix)=ntr
btg(ntr)%px=ix
btg(ntr)%pz=iz
!
! Now filter the new value up to its correct position
!
tpc=ntr
tpp=tpc/2
DO WHILE(tpp.gt.0)
   IF(ttn(iz,ix).lt.ttn(btg(tpp)%pz,btg(tpp)%px))THEN
      nsts(iz,ix)=tpp
      nsts(btg(tpp)%pz,btg(tpp)%px)=tpc
      exch=btg(tpc)
      btg(tpc)=btg(tpp)
      btg(tpp)=exch
      tpc=tpp
      tpp=tpc/2
   ELSE
      tpp=0
   ENDIF
ENDDO
END SUBROUTINE addtree

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! TYPE: SUBROUTINE
! CODE: FORTRAN 90
! This subroutine updates the binary tree after the root
! value has been used. The root is replaced by the value
! at the bottom of the tree, which is then filtered down
! to its correct position. This ensures that the tree remains
! balanced.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
SUBROUTINE downtree
IMPLICIT NONE
INTEGER :: tpp,tpc
REAL(KIND=i10) :: rd1,rd2
TYPE(backpointer) :: exch
!
! tpp = tree position of parent
! tpc = tree position of child
! exch = dummy to exchange btg values
! rd1,rd2 = substitution variables
!
! Replace root of tree with its last value
!
IF(ntr.EQ.1)THEN
   ntr=ntr-1
   RETURN
ENDIF
nsts(btg(ntr)%pz,btg(ntr)%px)=1
btg(1)=btg(ntr)
!
! Reduce size of tree by one
!
ntr=ntr-1
!
! Now filter new root down to its correct position
!
tpp=1
tpc=2*tpp
DO WHILE(tpc.lt.ntr)
!
! Check which of the two children is smallest - use the smallest
!
   rd1=ttn(btg(tpc)%pz,btg(tpc)%px)
   rd2=ttn(btg(tpc+1)%pz,btg(tpc+1)%px)
   IF(rd1.gt.rd2)THEN
      tpc=tpc+1
   ENDIF
!
!  Check whether the child is smaller than the parent; if so, then swap,
!  if not, then we are done
!
   rd1=ttn(btg(tpc)%pz,btg(tpc)%px)
   rd2=ttn(btg(tpp)%pz,btg(tpp)%px)
   IF(rd1.lt.rd2)THEN
      nsts(btg(tpp)%pz,btg(tpp)%px)=tpc
      nsts(btg(tpc)%pz,btg(tpc)%px)=tpp
      exch=btg(tpc)
      btg(tpc)=btg(tpp)
      btg(tpp)=exch
      tpp=tpc
      tpc=2*tpp
   ELSE
      tpc=ntr+1
   ENDIF
ENDDO
!
! If ntr is an even number, then we still have one more test to do
!
IF(tpc.eq.ntr)THEN
   rd1=ttn(btg(tpc)%pz,btg(tpc)%px)
   rd2=ttn(btg(tpp)%pz,btg(tpp)%px)
   IF(rd1.lt.rd2)THEN
      nsts(btg(tpp)%pz,btg(tpp)%px)=tpc
      nsts(btg(tpc)%pz,btg(tpc)%px)=tpp
      exch=btg(tpc)
      btg(tpc)=btg(tpp)
      btg(tpp)=exch
   ENDIF
ENDIF
END SUBROUTINE downtree

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! TYPE: SUBROUTINE
! CODE: FORTRAN 90
! This subroutine updates a value on the binary tree. The FMM
! should only produce updated values that are less than their
! prior values.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
SUBROUTINE updtree(iz,ix)
IMPLICIT NONE
INTEGER :: ix,iz,tpp,tpc
TYPE(backpointer) :: exch
!
! ix,iz = grid position of new addition to tree
! tpp = tree position of parent
! tpc = tree position of child
! exch = dummy to exchange btg values
!
! Filter the updated value to its correct position
!
tpc=nsts(iz,ix)
tpp=tpc/2
DO WHILE(tpp.gt.0)
   IF(ttn(iz,ix).lt.ttn(btg(tpp)%pz,btg(tpp)%px))THEN
      nsts(iz,ix)=tpp
      nsts(btg(tpp)%pz,btg(tpp)%px)=tpc
      exch=btg(tpc)
      btg(tpc)=btg(tpp)
      btg(tpp)=exch
      tpc=tpp
      tpp=tpc/2
   ELSE
      tpp=0
   ENDIF
ENDDO
END SUBROUTINE updtree

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! TYPE: SUBROUTINE
! CODE: FORTRAN 90
! This subroutine is passed four node values which lie on
! the corners of a rectangle and the coordinates of a point
! lying within the rectangle. It calculates the value at
! the internal point by using bilinear interpolation.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
SUBROUTINE bilinear(nv,dsx,dsz,biv)
USE globalp
IMPLICIT NONE
INTEGER :: i,j
REAL(KIND=i10) :: dsx,dsz,biv
REAL(KIND=i10), DIMENSION(2,2) :: nv
REAL(KIND=i10) :: produ
!
! nv = four node vertex values
! dsx,dsz = distance between internal point and top left node
! dnx,dnz = width and height of node rectangle
! biv = value at internal point calculated by bilinear interpolation
! produ = product variable
!
biv=0.0
DO i=1,2
   DO j=1,2
      produ=(1.0-ABS(((i-1)*dnx-dsx)/dnx))*(1.0-ABS(((j-1)*dnz-dsz)/dnz))
      biv=biv+nv(i,j)*produ
   ENDDO
ENDDO
END SUBROUTINE bilinear

END MODULE traveltime
