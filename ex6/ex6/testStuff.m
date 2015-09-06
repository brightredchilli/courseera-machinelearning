function stuff = testStuff()

 	a = magic(5);
 	a = a(:);

 	for i: a
 		fprintf(i);
 	end

	stuff = 0;

end
